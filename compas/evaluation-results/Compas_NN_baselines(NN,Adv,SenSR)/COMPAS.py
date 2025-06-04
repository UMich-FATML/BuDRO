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
import AdvDebCustom
import data_pre
import script
from itertools import product

tf.compat.v1.disable_eager_execution()


def get_compas_orig():
    '''
    Preprocess the compas data set by removing some features and put compas data into a BinaryLabelDataset
    You can load the compas dataset from aif360.datasets
    '''
    dataset_orig = data_pre.load_preproc_data_compas()
    display(Markdown("#### Dataset shape"))
    print(dataset_orig.features.shape)
    display(Markdown("#### Dataset feature names"))
    print(dataset_orig.feature_names)
    dataset_orig.features = dataset_orig.features[:,:-1]

    return dataset_orig


def preprocess_compas_data(seed = None):
    '''
    Description: Ths code (1) standardizes the continuous features, (2) one hot encodes the categorical features, (3) splits into a train (80%) and test set (20%), (4) based on this data, create another copy where race is deleted as a predictive feature and the feature we predict is race (learning the sensitive directions)
    Input: seed: the seed used to split data into train/test
    '''
    # Get the dataset 
    dataset_orig = get_compas_orig()

    # we will standardize continous features
    continous_features = ['priors_count']
    continous_features_indices = [dataset_orig.feature_names.index(feat) for feat in continous_features]

    # Get the dataset and split into train, valid and test
    dataset_orig_train, dataset_orig_test = dataset_orig.split([0.8], shuffle=True)
    
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

    # Also create a train/test set where the predictive features (X) do not include race and race is what you want to predict (y). This is used when learnng the sensitive direction for SenSR
    X_race_train = np.delete(X_train, [dataset_orig_test.feature_names.index(feat) for feat in ['race']], axis = 1)
    X_race_test = np.delete(X_test, [dataset_orig_test.feature_names.index(feat) for feat in ['race']], axis = 1)

    y_race_train = dataset_orig_train.features[:, dataset_orig_train.feature_names.index('race')]
    y_race_test = dataset_orig_test.features[:, dataset_orig_test.feature_names.index('race')]

    return X_train, X_test, y_train, y_test, X_race_train, X_race_test, y_race_train, y_race_test, dataset_orig_train, dataset_orig_test


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


def compute_balanced_accuracy(dataset_orig, preds):
    '''
    Description: computes the balanced accuracy, i.e. the average of TPR and TNR
    Input: data_set: a BinaryLabelDataset (from the aif360 module)
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
    
    TPR = classified_metric.true_positive_rate()
    TNR = classified_metric.true_negative_rate()
    return 0.5*(TPR+TNR)


def get_consistency(X, weights=0, dataset_orig_test = 0, adv = 0):
    '''
    Description: Ths function computes gender and race consistency.
    Input:
        X: numpy matrix of predictive features
        weights: learned weights for project, baseline, and sensr
        adv: the adversarial debiasing object if using adversarial Adversarial Debiasing
        dataset_orig_test: this is the data in a BinaryLabelDataset format when using adversarial debiasing
    '''
#     gender_race_idx = [gender_idx, race_idx]
    gender_idx = 0
    race_idx = 1
    if adv == 0:
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
        
        if adv == 0:
            X0_logits = l_pred.eval(feed_dict={tf_X: X0})
            X0_preds = np.argmax(X0_logits, axis = 1)
        else:
            dataset_mod = dataset_orig_test.copy(deepcopy=True)
            dataset_mod.features = X0
            dataset_mod, _ = adv.predict(dataset_mod)
            X0_preds = np.asarray([i[0] for i in dataset_mod.labels])

        ## copy 2
        X1 = np.copy(X)
        X1[:, race_idx] = 1
        
        if adv == 0:
            X1_logits = l_pred.eval(feed_dict={tf_X: X1})
            X1_preds = np.argmax(X1_logits, axis = 1)
        else:
            dataset_mod = dataset_orig_test.copy(deepcopy=True)
            dataset_mod.features = X1
            dataset_mod, _ = adv.predict(dataset_mod)
            X1_preds = np.asarray([i[0] for i in dataset_mod.labels])
        
        race_consistency =  (X0_preds*X1_preds).sum() + ((1-X0_preds)*(1-X1_preds)).sum()
        race_consistency /= n

        ### copy 3
        Xg0 = np.copy(X)
        Xg0[:, gender_idx] = 0
        
        if adv == 0:
            Xg0_logits = l_pred.eval(feed_dict={tf_X: Xg0})
            Xg0_preds = np.argmax(Xg0_logits, axis = 1)
        else:
            dataset_mod = dataset_orig_test.copy(deepcopy=True)
            dataset_mod.features = Xg0
            dataset_mod, _ = adv.predict(dataset_mod)
            Xg0_preds =np.asarray( [i[0] for i in dataset_mod.labels])

        #### copy 4
        Xg1 = np.copy(X)
        Xg1[:, gender_idx] = 1
        
        if adv == 0:
            Xg1_logits = l_pred.eval(feed_dict={tf_X: Xg1})
            Xg1_preds = np.argmax(Xg1_logits, axis = 1)
        else:
            dataset_mod = dataset_orig_test.copy(deepcopy=True)
            dataset_mod.features = Xg1
            dataset_mod, _ = adv.predict(dataset_mod)
            Xg1_preds = np.asarray([i[0] for i in dataset_mod.labels])
        
        gender_consistency =  (Xg0_preds*Xg1_preds).sum() + ((1-Xg0_preds)*(1-Xg1_preds)).sum()
        gender_consistency /= n

        return race_consistency, gender_consistency


def run_baseline_experiment(X_train, y_train, X_test, y_test, lr):
    return SenSR.train_nn(X_train, y_train, X_test = X_test, y_test = y_test, n_units=[100], l2_reg=0., lr = lr, batch_size=1000, epoch=27000, verbose=True)

def run_SenSR_experiment(seed, row_list, dataset_orig_test, sind, rind, X_train, X_test, X_race_train, X_race_test, y_train, y_test, y_race_train, y_race_test, eps, fe, flr, se, slr):
    
    feature_names = dataset_orig_test.feature_names
    sensitive_directions = compas_proj_matrix_sensr(X_train, rind, feature_names)
    
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
        epoch=2000,
        verbose=True,
        l2_reg=0.,
        lamb_init=2.,
        subspace_epoch=se,
        subspace_step=slr,
        eps=eps,
        full_step=flr,
        full_epoch=fe)


def run_adv_deb_experiment(dataset_orig_train, dataset_orig_test):
    sess = tf.compat.v1.Session()
    tf.compat.v1.name_scope("my_scope")
    privileged_groups = [{'race':1}]
    unprivileged_groups = [{'race':0}]
    adv = AdvDebCustom.AdversarialDebiasing(unprivileged_groups, privileged_groups, "my_scope", sess, seed=None, adversary_loss_weight=0.001, num_epochs=100, batch_size=200, classifier_num_hidden_units=1200, debias=True)

    _ = adv.fit(dataset_orig_train)
    test_data, _ = adv.predict(dataset_orig_test)

    return adv, test_data.labels

def run_experiments(name, seed):
    '''
    Description: Scripts for running baselines based on neural network.
    Inputs: 
    name: name of the experiment. Valid choices are vanila neural network (NN), SenSR, adv_deb
    seed: train/test splitting seed
    Outputs:
    Saving all evaluation metrics associated with hyper-parameters to a csv file
    '''

    if name not in ['NN', 'SenSR', 'adv_deb']:
        raise ValueError('You did not specify a valid experiment to run.')

    # Loading train/test splitting seeds
    seeds = seed
    # List for saving evaluation metrics
    row_list = []
    
    ### Parameters for NN  ###
    # An example for parameters: 
    lr = 5e-6    # general parameter learning rate 
    if name == 'SenSR':
        fe = 10
        se = 10
        eps = 0.01
        flr = 0.001
        slr = 0.1
    elif name == 'adv_deb':
        adv_weight = 0.1
    
    for seed in seeds:
        print('On experiment', seed)

        # get train/test data
        np.random.seed(seed)
        X_train, X_test, y_train, y_test, X_race_train, X_race_test, y_race_train, y_race_test, dataset_orig_train, dataset_orig_test = preprocess_compas_data(seed = seed)
        
        # Get binary gender and race features
        sind = dataset_orig_train.feature_names.index('sex')
        rind = dataset_orig_train.feature_names.index('race')
        y_sex_test = X_test[:, sind]
        y_race_test = X_test[:, rind]

        tf.compat.v1.reset_default_graph()

        # run neural network
        if name == 'NN':
            weights, train_logits, test_logits  = run_baseline_experiment(X_train, y_train, X_test, y_test, lr)
        elif name == 'SenSR':
            rows_list, weights, train_logits, test_logits = run_SenSR_experiment(seed, row_list, dataset_orig_test, sind, rind, X_train, X_test, X_race_train, X_race_test, y_train, y_test, y_race_train, y_race_test, eps, fe, flr, se, slr)
        elif name == 'adv_deb':
            adv, preds = run_adv_deb_experiment(dataset_orig_train, dataset_orig_test)
        
        ###Calculating all evaluation metrics###
        # get race and gender consistency
        if name == 'adv_deb':
            race_consistency, gender_consistency = get_consistency(X_test, dataset_orig_test = dataset_orig_test, adv = adv)
        else:
            race_consistency, gender_consistency = get_consistency(X_test, weights = weights)

        if name != 'adv_deb':
            preds = np.argmax(test_logits, axis = 1)
        
        # Calculate other evaluation metrics
        bal_acc_temp = compute_balanced_accuracy(dataset_orig_test, preds)
        yt = dataset_orig_test.labels
        y_guess = preds
        if name != 'adv_deb':
            yt = np.reshape(yt, (-1, ))
        base_sex = script.group_metrics(yt, y_guess, y_sex_test, label_good=1)
        base_race = script.group_metrics(yt, y_guess, y_race_test, label_good=1)
        
        print('Saving results ......')
        save_data = {}
        save_data['seed'] =  seed
        if name == 'adv_deb':
            save_data['acc'] = base_sex[-1][0]
        else:
            save_data['acc'] = base_sex[-1]
        save_data['bl_acc'] = bal_acc_temp
        save_data['gcons'] = gender_consistency
        save_data['rcons'] = race_consistency 
        save_data['RMS(G)'] = base_sex[4]
        save_data['MAX(G)'] = base_sex[5]
        save_data['AOD(G)'] = base_sex[6]
        save_data['EOD(G)'] = base_sex[7]
        save_data['SPD(G)'] = base_sex[8]
        save_data['RMS(R)'] = base_race[4]
        save_data['MAX(R)'] = base_race[5]
        save_data['AOD(R)'] = base_race[6]
        save_data['EOD(R)'] = base_race[7]
        save_data['SPD(R)'] = base_race[8]
        row_list.append(save_data)
    
    # Saving results to a csv file
    dfsave_data = pd.DataFrame(data=row_list)
    filename = 'result:' + 'seed_%d' % seed + '_' + name + '_Compas.csv'
    dfsave_data.to_csv(filename, index=True)
    
    print('DONE')
    return 0

# Run experiments
seed = [1, 6, 8, 26, 27, 30, 36, 47, 63, 70]
run_experiments('adv_deb', seed) # for neural network baseline with random seed 1
