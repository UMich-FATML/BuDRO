import numpy as np
import pandas as pd
from sklearn import preprocessing
from aif360.datasets import BinaryLabelDataset
import tensorflow as tf
import time
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy.stats import sem
import SenSR
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from sklearn.decomposition import TruncatedSVD
from IPython.display import display, Markdown, Latex
from sklearn.linear_model import LogisticRegressionCV

import data_pre
import script
from itertools import product


tf.compat.v1.disable_eager_execution()


def get_compas_orig():

    dataset_orig = data_pre.load_preproc_data_compas()
    display(Markdown("#### Dataset shape"))
    print(dataset_orig.features.shape)
    display(Markdown("#### Dataset feature names"))
    print(dataset_orig.feature_names)
    dataset_orig.features = dataset_orig.features[:,:-1]
    display(Markdown("#### Favorable and unfavorable labels"))
    print(dataset_orig.favorable_label, dataset_orig.unfavorable_label)
    display(Markdown("#### Protected attribute names"))
    print(dataset_orig.protected_attribute_names)
    display(Markdown("#### Privileged and unprivileged protected attribute values"))
    print(dataset_orig.privileged_protected_attributes, dataset_orig.unprivileged_protected_attributes)

    return dataset_orig


def preprocess_compas_data(seed = None):
    '''
    Description: Ths code (1) standardizes the continuous features, (2) one hot encodes the categorical features, (3) splits into a train (80%) and test set (20%), (4) based on this data, create another copy where gender is deleted as a predictive feature and the feature we predict is gender (used by SenSR when learning the sensitive directions)

    Input: seed: the seed used to split data into train/test
    '''
    # Get the dataset and split into train and test
    dataset_orig = get_compas_orig()

    # we will standardize continous features
    continous_features = ['priors_count']
    continous_features_indices = [dataset_orig.feature_names.index(feat) for feat in continous_features]

    # get a 80%/20% train/test split
    dataset_orig_train, dataset_orig_test = dataset_orig.split([0.8], shuffle=True, seed = seed)
    SS = StandardScaler().fit(dataset_orig_train.features[:, continous_features_indices])
    dataset_orig_train.features[:, continous_features_indices] = SS.transform(dataset_orig_train.features[:, continous_features_indices])
    dataset_orig_test.features[:, continous_features_indices] = SS.transform(dataset_orig_test.features[:, continous_features_indices])

    X_train = dataset_orig_train.features
    X_test = dataset_orig_test.features

    y_train = dataset_orig_train.labels
    y_test = dataset_orig_test.labels
    
    one_hot = OneHotEncoder(sparse=False)
    one_hot.fit(y_train.reshape(-1,1))
    names_income = one_hot.categories_
    y_train = one_hot.transform(y_train.reshape(-1,1))
    y_test = one_hot.transform(y_test.reshape(-1,1))
#     print('y_train2:', y_train.shape)
    

    # Also create a train/test set where the predictive features (X) do not include gender and gender is what you want to predict (y). This is used when learnng the sensitive direction for SenSR
    X_gender_train = np.delete(X_train, [dataset_orig_test.feature_names.index(feat) for feat in ['race']], axis = 1)
    X_gender_test = np.delete(X_test, [dataset_orig_test.feature_names.index(feat) for feat in ['race']], axis = 1)

    y_gender_train = dataset_orig_train.features[:, dataset_orig_train.feature_names.index('race')]
    y_gender_test = dataset_orig_test.features[:, dataset_orig_test.feature_names.index('race')]

    one_hot.fit(y_gender_train.reshape(-1,1))
    names_gender = one_hot.categories_
    y_gender_train = one_hot.transform(y_gender_train.reshape(-1,1))
    y_gender_test = one_hot.transform(y_gender_test.reshape(-1,1))

    return X_train, X_test, y_train, y_test, X_gender_train, X_gender_test, y_gender_train, y_gender_test, dataset_orig_train, dataset_orig_test

def save_to_file(directory, variable, name):
    timestamp = str(int(time.time()))
    with open(directory + name + '_' + timestamp + '.txt', "w") as f:
        f.write(str(np.mean(variable))+"\n")
        f.write(str(sem(variable))+"\n")
        for s in variable:
            f.write(str(s) +"\n")

def compute_gap_RMS_and_gap_max(data_set):
    '''
    Description: computes the gap RMS and max gap
    Input: data_set: a BinaryLabelDataset (from the aif360 module)
    '''
    TPR = -1*data_set.false_negative_rate_difference()
    TNR = -1*data_set.false_positive_rate_difference()

    return np.sqrt(1/2*(TPR**2 + TNR**2)), max(np.abs(TPR), np.abs(TNR))

def compute_balanced_accuracy(data_set):
    '''
    Description: computes the balanced accuracy, i.e. the average of TPR and TNR
    Input: data_set: a BinaryLabelDataset (from the aif360 module)
    '''
    TPR = data_set.true_positive_rate()
    TNR = data_set.true_negative_rate()
    return 0.5*(TPR+TNR)

def get_consistency(X, weights=0):
    '''
    Description: Ths function computes spouse consistency and gender and race consistency.
    Input:
        X: numpy matrix of predictive features
        weights: learned weights for project, baseline, and sensr
        proj: if using the project first baseline, this is the projection matrix
        gender_idx: column corresponding to the binary gender variable
        race_idx: column corresponding to the binary race variable
        relationship)_idx: list of column for the following features: relationship_ Husband, relationship_ Not-in-family, relationship_ Other-relative, relationship_ Own-child, relationship_ Unmarried, relationship_ Wife
        husband_idx: column corresponding to the husband variable
        wife_idx: column corresponding to the wife variable
        adv: the adversarial debiasing object if using adversarial Adversarial Debiasing
        dataset_orig_test: this is the data in a BinaryLabelDataset format when using adversarial debiasing
    '''
#     gender_race_idx = [gender_idx, race_idx]
    gender_idx = 0
    race_idx = 1
    N, D = X.shape
    K = 1

    tf_X = tf.compat.v1.placeholder(tf.float32, shape=[None,D])
    tf_y = tf.compat.v1.placeholder(tf.float32, shape=[None,K], name='response')

    n_units = weights[1].shape
    n_units = n_units[0]

    _, l_pred, _, _ = SenSR.forward(tf_X, tf_y, weights=weights, n_units = n_units, activ_f = tf.nn.relu)
    
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        n, _ = X.shape

        # make 4 versions of the original data by changing binary gender and gender, then count how many classifications change
        # copy 1
        X0 = np.copy(X)
        X0[:, race_idx] = 0

        X0_logits = l_pred.eval(feed_dict={tf_X: X0})
        X0_preds = np.argmax(X0_logits, axis = 1)

        ## copy 2
        X1 = np.copy(X)
        X1[:, race_idx] = 1

        X1_logits = l_pred.eval(feed_dict={tf_X: X1})
        X1_preds = np.argmax(X1_logits, axis = 1)
        
        race_consistency =  (X0_preds*X1_preds).sum() + ((1-X0_preds)*(1-X1_preds)).sum()
        race_consistency /= n

        ### copy 3
        Xg0 = np.copy(X)
        Xg0[:, gender_idx] = 0

        Xg0_logits = l_pred.eval(feed_dict={tf_X: Xg0})
        Xg0_preds = np.argmax(Xg0_logits, axis = 1)

        #### copy 4
        Xg1 = np.copy(X)
        Xg1[:, gender_idx] = 1

        Xg1_logits = l_pred.eval(feed_dict={tf_X: Xg1})
        Xg1_preds = np.argmax(Xg1_logits, axis = 1)
        
        gender_consistency =  (Xg0_preds*Xg1_preds).sum() + ((1-Xg0_preds)*(1-Xg1_preds)).sum()
        gender_consistency /= n

        return race_consistency, gender_consistency

def get_sensitive_directions_and_projection_matrix(X_gender_train, y_gender_train, X_gender_test, y_gender_test, gender_race_features_idx = [0, 1] ):
    '''
    Description: Get the sensitive directions and projection matrix. he sensitive directions include the race and gender direction as well as the learned hyperplane that predicts gender (without using gender as a predictive feature of course).
    '''
    weights, train_logits, test_logits = SenSR.train_nn(X_gender_train, y_gender_train, X_test = X_gender_test, y_test = y_gender_test, n_units=[], l2_reg=.1, batch_size=1000, epoch=1000, verbose=True)

    n, d = weights[0].shape
    sensitive_directions = []
    full_weights = np.zeros((n+1,d))
    full_weights[0:n-1,:] = weights[0][0:n-1,:]
    full_weights[n, :] = weights[0][n-1,:]
    sensitive_directions.append(full_weights.T)

    for idx in gender_race_features_idx:
        temp_direction = np.zeros((n+1,1)).reshape(1,-1)
        temp_direction[0, idx] = 1
        sensitive_directions.append(np.copy(temp_direction))

    sensitive_directions = np.vstack(sensitive_directions)
    tSVD = TruncatedSVD(n_components= 2 + len(gender_race_features_idx))
    tSVD.fit(sensitive_directions)
    sensitive_directions = tSVD.components_

    return sensitive_directions, SenSR.compl_svd_projector(sensitive_directions)


def compas_feature_logreg(
        X_train,
        rind=None,
        feature_names=None,
        labels=None
        ):
    X_test = None
    if feature_names and rind:
        print("Fitting LR to feature {}".format(feature_names[rind]))

    LR = LogisticRegressionCV(Cs=100, cv=5, max_iter=5000)

    # data for training logistic regression
    if X_test is not None:
        XLR = np.vstack((X_train, X_test))
    else:
        XLR = np.copy(X_train)

    if rind is not None: 
        targets = XLR[:,rind].copy()
        XLR[:,rind] = np.zeros(XLR.shape[0])

    elif labels is not None: 
        targets = labels

    else:
        print("Error: no labels provided for logistic regression")
        print("Error: no model trained", flush=True)
        return LR

    LR.fit( XLR, targets )

    return LR

# projection onto sensitive subspace
def compas_proj_matrix_sensr(X_train, rind, feature_names):
    
    eg = np.zeros(X_train.shape[1])
    eg[feature_names.index('sex')] = 1.0

    er = np.zeros(X_train.shape[1])
    er[feature_names.index('race')] = 1.0

    TLR = compas_feature_logreg(X_train, rind=rind)
    wg = TLR.coef_[0]

    A = np.array([wg, eg, er])
    save=False

    return A
#     return script.proj_matrix_gen(X_train, A, save)


def get_metrics(dataset_orig, preds):
    '''
    Description: This code computes accuracy, balanced accuracy, max gap and gap rms for race and gender
    Input: dataset_orig: a BinaryLabelDataset (from the aif360 module)
            preds: predictions
    '''
    dataset_learned_model = dataset_orig.copy()
    dataset_learned_model.labels = preds

    # wrt gender
    privileged_groups = [{'sex': 1}]
    unprivileged_groups = [{'sex': 0}]

    classified_metric = ClassificationMetric(dataset_orig,
                                                     dataset_learned_model,
                                                     unprivileged_groups=unprivileged_groups,
                                                     privileged_groups=privileged_groups)

    bal_acc = compute_balanced_accuracy(classified_metric)

    gender_gap_rms, gender_max_gap = compute_gap_RMS_and_gap_max(classified_metric)
    print("Test set: gender gap rms = %f" % gender_gap_rms)
    print("Test set: gender max gap rms = %f" % gender_max_gap)
    print("Test set: Balanced TPR = %f" % bal_acc)
    

    # wrt race
    privileged_groups = [{'race': 1}]
    unprivileged_groups = [{'race': 0}]
    classified_metric = ClassificationMetric(dataset_orig,
                                                     dataset_learned_model,
                                                     unprivileged_groups=unprivileged_groups,
                                                     privileged_groups=privileged_groups)

    race_gap_rms, race_max_gap = compute_gap_RMS_and_gap_max(classified_metric)
    print("Test set: race gap rms = %f" % race_gap_rms)
    print("Test set: race max gap rms = %f" % race_max_gap)


    return bal_acc, classified_metric.accuracy(), race_gap_rms, race_max_gap, gender_gap_rms, gender_max_gap

def run_baseline_experiment(X_train, y_train, X_test, y_test):
    return SenSR.train_nn(X_train, y_train, X_test = X_test, y_test = y_test, n_units=[100], l2_reg=0., lr = .00001, batch_size=1000, epoch=10, verbose=True)

def run_SenSR_experiment(seed, row_list, dataset_orig_test, sind, rind, X_train, X_test, X_gender_train, X_gender_test, y_train, y_test, y_gender_train, y_gender_test, eps, fe, flr, se, slr):
    
    feature_names = dataset_orig_test.feature_names
    sensitive_directions = compas_proj_matrix_sensr(X_train, rind, feature_names)
#     print('sensitive_directions type:', type(sensitive_directions))
#     print('sensitive_directions shape', sensitive_directions.shape)
    
    return SenSR.train_fair_nn(
        seed, row_list, 
        dataset_orig_test, sind, rind,
        X_train,
        y_train,
        sensitive_directions,
        X_test=X_test,
        y_test=y_test,
        n_units = [100],
        lr=.0001,
        batch_size=1000,
        epoch=4000,
        verbose=True,
        l2_reg=0.,
        lamb_init=2.,
        subspace_epoch=se,
        subspace_step=slr,
        eps=eps,
        full_step=flr,
        full_epoch=fe)

def run_project_experiment(X_train, X_test, X_gender_train, X_gender_test, y_train, y_test, y_gender_train, y_gender_test, directory, i):
    _, proj_compl = get_sensitive_directions_and_projection_matrix(X_gender_train, y_gender_train, X_gender_test, y_gender_test)

    np.save(directory+'proj_compl_'+str(i), proj_compl)

    X_train_proj = X_train@proj_compl
    X_test_proj = X_test@proj_compl
    weights, train_logits, test_logits = SenSR.train_nn(
        X_train_proj,
        y_train,
        X_test = X_test_proj,
        y_test = y_test,
        n_units=[100],
        l2_reg=0.,
        lr = .00001,
        batch_size=1000,
        epoch=60000,
        verbose=False)
    return weights, train_logits, test_logits, proj_compl

def run_adv_deb_experiment(dataset_orig_train, dataset_orig_test):
    sess = tf.compat.v1.Session()
    tf.compat.v1.name_scope("my_scope")
    privileged_groups = [{'sex_ Male': 1, 'race_ White':1}]
    unprivileged_groups = [{'sex_ Male': 0, 'race_ White':0}]
    adv = AdvDebCustom.AdversarialDebiasing(unprivileged_groups, privileged_groups, "my_scope", sess, seed=None, adversary_loss_weight=0.001, num_epochs=500, batch_size=1000, classifier_num_hidden_units=100, debias=True)

    _ = adv.fit(dataset_orig_train)
    test_data, _ = adv.predict(dataset_orig_test)

    return adv, test_data.labels

def run_experiments(name):
    '''
    Description: Run each experiment num_exp times where a new train/test split is generated. Save results in the path specified by directory

    Inputs: name: name of the experiment. Valid choices are baseline, project, SenSR, adv_deb
    '''

    if name not in ['baseline', 'project', 'SenSR', 'adv_deb']:
        raise ValueError('You did not specify a valid experiment to run.')

    gender_race_consistencies = []
    spouse_consistencies = []

    accuracy = []
    balanced_accuracy = []

    gender_gap_rms = []
    gender_max_gap = []

    race_gap_rms = []
    race_max_gap = []
    seeds = [6]
    row_list = []
    
    param = {
    'eps': 0.001,
    'fe': 1000,
    'flr': 0.001,
    'se': 1.0,
    'slr': 50.0
    }
    
    epss = 0.001
    flr = 0.0001
    slr = 0.1
    eps_grid = [epss*1, epss*2, epss*5, epss*8, epss*10, epss*15, epss*20, epss*50, epss*80, epss*100]
    fe_grid = [10]
    flr_grid = [0.1*flr, flr, flr*2, flr*5, flr*10]
    se_grid = [10]
    slr_grid = [0.1*slr, slr, 10*slr]
    hypers = [eps_grid, fe_grid, flr_grid, se_grid, slr_grid]
    names = ['eps', 'fe', 'flr', 'se', 'slr']    
    
    for seed in seeds:
        print('On experiment', seed)

        # get train/test data
        X_train, X_test, y_train, y_test, X_gender_train, X_gender_test, y_gender_train, y_gender_test, dataset_orig_train, dataset_orig_test = preprocess_compas_data(seed = seed)
        
        sind = dataset_orig_train.feature_names.index('sex')
        rind = dataset_orig_train.feature_names.index('race')
        y_sex_test = X_test[:, sind]
        y_race_test = X_test[:, rind]
        
        for pack in product(*hypers):
            values = list(pack)

            for ele in range(len(values)):
                param[names[ele]] = values[ele]
            print(param)
            eps = param['eps']
            fe = param['fe']
            flr = param['flr']
            se = param['se']
            slr = param['slr']


            tf.compat.v1.reset_default_graph()

            # run experiments
            if name == 'baseline':
                weights, train_logits, test_logits  = run_baseline_experiment(X_train, y_train, X_test, y_test)
            elif name == 'SenSR':
                rows_list, weights, train_logits, test_logits = run_SenSR_experiment(seed, row_list, dataset_orig_test, sind, rind, X_train, X_test, X_gender_train, X_gender_test, y_train, y_test, y_gender_train, y_gender_test, eps, fe, flr, se, slr)
            elif name == 'project':
                weights, train_logits, test_logits, proj_compl = run_project_experiment(X_train, X_test, X_gender_train, X_gender_test, y_train, y_test, y_gender_train, y_gender_test)
            elif name == 'adv_deb':
                adv, preds = run_adv_deb_experiment(dataset_orig_train, dataset_orig_test)

            # get race/gender and spouse consistency

            if name == 'project':
                gender_race_consistency = get_consistency(X_test, weights = weights, proj = proj_compl)
            elif name == 'adv_deb':
                gender_race_consistency = get_consistency(X_test, adv = adv, dataset_orig_test = dataset_orig_test)
            else:
                race_consistency, gender_consistency = get_consistency(X_test, weights = weights)
    #         gender_race_consistencies.append(gender_race_consistency)
            print('race_consistency', race_consistency)
            print('gender_consistency', gender_consistency)


    #         if name != 'adv_deb':
    #             np.save(directory+'weights_'+str(i), weights)
            preds = np.argmax(test_logits, axis = 1)
            bal_acc_temp, acc_temp, race_gap_rms_temp, race_max_gap_temp, gender_gap_rms_temp, gender_max_gap_temp = get_metrics(dataset_orig_test, preds)
            yt = dataset_orig_test.labels
            yt = np.reshape(yt, (-1, ))
            y_guess = preds
            print("SEX")
            base_sex = script.group_metrics(yt, y_guess, y_sex_test, label_good=1)
            print("RACE")
            base_race = script.group_metrics(yt, y_guess, y_race_test, label_good=1)

    dfsave_data = pd.DataFrame(data=rows_list)
    dfsave_data.to_csv('ave.csv', index=True)
    print('DONE')
    
    return 0

run_experiments('SenSR')
