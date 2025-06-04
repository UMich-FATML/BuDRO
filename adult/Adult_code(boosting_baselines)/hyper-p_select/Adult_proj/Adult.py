import numpy as np
import pandas as pd
import os

from sklearn import preprocessing
from aif360.datasets import BinaryLabelDataset
import tensorflow as tf
import time
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy.stats import sem
import SenSR2 as SenSR
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from sklearn.decomposition import TruncatedSVD
# import AdvDebCustom
from IPython.display import display, Markdown, Latex
from sklearn.linear_model import LogisticRegressionCV
from copy import deepcopy

import data_pre
import script
from itertools import product

import xgboost as xgb


tf.compat.v1.disable_eager_execution()

def get_adult_data():
    '''
    Preprocess the adult data set by removing some features and put adult data into a BinaryLabelDataset
    You need to download the adult dataset (both the adult.data and adult.test files) from https://archive.ics.uci.edu/ml/datasets/Adult
    '''

    headers = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-stataus', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'y']
    train = pd.read_csv('adult.data', header = None)
    test = pd.read_csv('adult.test', header = None)
    df = pd.concat([train, test], ignore_index=True)
    df.columns = headers

    df['y'] = df['y'].replace({' <=50K.': 0, ' >50K.': 1, ' >50K': 1, ' <=50K': 0 })

    df = df.drop(df[(df[headers[-2]] == ' ?') | (df[headers[6]] == ' ?')].index)
    df = pd.get_dummies(df, columns=[headers[1], headers[5], headers[6], headers[7], headers[9], headers[8], 'native-country'])

    delete_these = ['race_ Amer-Indian-Eskimo','race_ Asian-Pac-Islander','race_ Black','race_ Other', 'sex_ Female']

    delete_these += ['native-country_ Cambodia', 'native-country_ Canada', 'native-country_ China', 'native-country_ Columbia', 'native-country_ Cuba', 'native-country_ Dominican-Republic', 'native-country_ Ecuador', 'native-country_ El-Salvador', 'native-country_ England', 'native-country_ France', 'native-country_ Germany', 'native-country_ Greece', 'native-country_ Guatemala', 'native-country_ Haiti', 'native-country_ Holand-Netherlands', 'native-country_ Honduras', 'native-country_ Hong', 'native-country_ Hungary', 'native-country_ India', 'native-country_ Iran', 'native-country_ Ireland', 'native-country_ Italy', 'native-country_ Jamaica', 'native-country_ Japan', 'native-country_ Laos', 'native-country_ Mexico', 'native-country_ Nicaragua', 'native-country_ Outlying-US(Guam-USVI-etc)', 'native-country_ Peru', 'native-country_ Philippines', 'native-country_ Poland', 'native-country_ Portugal', 'native-country_ Puerto-Rico', 'native-country_ Scotland', 'native-country_ South', 'native-country_ Taiwan', 'native-country_ Thailand', 'native-country_ Trinadad&Tobago', 'native-country_ United-States', 'native-country_ Vietnam', 'native-country_ Yugoslavia']

    delete_these += ['fnlwgt', 'education']

    df.drop(delete_these, axis=1, inplace=True)

    return BinaryLabelDataset(df = df, label_names = ['y'], protected_attribute_names = ['sex_ Male', 'race_ White'])

def get_adult_orig():
#     # Way 1:
#     dataset_orig = load_preproc_data_compas()
#     # print out some labels, names, etc.
#     display(Markdown("#### Dataset shape"))
#     print(dataset_orig.features.shape)
#     display(Markdown("#### Dataset feature names"))
#     print(dataset_orig.feature_names)
#     dataset_orig.features = dataset_orig.features[:,:-1]

    # Way 2: New dataset
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


def preprocess_adult_data(seed = 0):
    '''
    Description: Ths code (1) standardizes the continuous features, (2) one hot encodes the categorical features, (3) splits into a train (80%) and test set (20%), (4) based on this data, create another copy where gender is deleted as a predictive feature and the feature we predict is gender (used by SenSR when learning the sensitive directions)

    Input: seed: the seed used to split data into train/test
    '''
    # Get the dataset and split into train and test
    dataset_orig = get_adult_data()

    # we will standardize continous features
    continous_features = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
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

    # Also create a train/test set where the predictive features (X) do not include gender and gender is what you want to predict (y). This is used when learnng the sensitive direction for SenSR
    X_gender_train = np.delete(X_train, [dataset_orig_test.feature_names.index(feat) for feat in ['sex_ Male']], axis = 1)
    X_gender_test = np.delete(X_test, [dataset_orig_test.feature_names.index(feat) for feat in ['sex_ Male']], axis = 1)

    y_gender_train = dataset_orig_train.features[:, dataset_orig_train.feature_names.index('sex_ Male')]
    y_gender_test = dataset_orig_test.features[:, dataset_orig_test.feature_names.index('sex_ Male')]

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

def get_consistency_bst(X, bst=0, proj = 0, gender_idx = 39, race_idx = 40, relationship_idx = [33, 34, 35, 36, 37, 38], husband_idx = 33, wife_idx = 38, adv = 0, dataset_orig_test = 0):
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
    gender_race_idx = [gender_idx, race_idx]

    if adv == 0:
        N, D = X.shape
        K = 1

#         tf_X = tf.compat.v1.placeholder(tf.float32, shape=[None,D])
#         tf_y = tf.compat.v1.placeholder(tf.float32, shape=[None,K], name='response')

#         n_units = weights[1].shape
#         n_units = n_units[0]

#         _, l_pred, _, _ = SenSR.forward(tf_X, tf_y, weights=weights, n_units = n_units, activ_f = tf.nn.relu)

#     with tf.compat.v1.Session() as sess:
#         sess.run(tf.compat.v1.global_variables_initializer())
        n, _ = X.shape

        # make 4 versions of the original data by changing binary gender and gender, then count how many classifications change
        #copy 1
        X00 = np.copy(X)
        X00[:, gender_race_idx] = 0

        if np.ndim(proj) != 0:
            
            X00 = X00@proj

        preds00 = bst.predict(xgb.DMatrix(X00))
        y00 = np.array([ 1 if pval > 0.5 else 0 for pval in preds00])

        #### copy 2
        X01 = np.copy(X)
        X01[:, gender_race_idx] = 0
        X01[:, gender_idx] = 1

        if np.ndim(proj) != 0:
            X01 = X01@proj

        preds01 = bst.predict(xgb.DMatrix(X01))
        y01 = np.array([ 1 if pval > 0.5 else 0 for pval in preds01])

        #### copy 3
        X10 = np.copy(X)
        X10[:, gender_race_idx] = 0
        X10[:, race_idx] = 1

        if np.ndim(proj) != 0:
            X10 = X10@proj

        preds10 = bst.predict(xgb.DMatrix(X10))
        y10 = np.array([ 1 if pval > 0.5 else 0 for pval in preds10])

        #### copy 4
        X11 = np.copy(X)
        X11[:, race_idx] = 1
        X11[:, gender_idx] = 1

        if np.ndim(proj) != 0:
            X11 = X11@proj

        preds11 = bst.predict(xgb.DMatrix(X11))
        y11 = np.array([ 1 if pval > 0.5 else 0 for pval in preds11])

        gender_and_race_consistency = np.mean([1 if y00[i] == y01[i] and y00[i] == y10[i] and y00[i] == y11[i] else 0 for i in range(len(y00))])
#         gender_and_race_consistency = np.mean([1 if X00_preds[i] == X01_preds[i] and X00_preds[i] == X10_preds[i] and X00_preds[i] == X11_preds[i] else 0 for i in range(len(X00_preds))])

        # make two copies of every datapoint: one which is a husband and one which is a wife. Then count how many classifications change
        X_husbands = np.copy(X)
        X_husbands[:,relationship_idx] = 0
        X_husbands[:,husband_idx] = 1

        if np.ndim(proj) != 0:
            X_husbands = X_husbands@proj

        predshs = bst.predict(xgb.DMatrix(X_husbands))
        yhs = np.array([ 1 if pval > 0.5 else 0 for pval in predshs])

        X_wives = np.copy(X)
        X_wives[:,relationship_idx] = 0
        X_wives[:,wife_idx] = 1

        if np.ndim(proj) != 0:
            X_wives = X_wives@proj

        predswf = bst.predict(xgb.DMatrix(X_wives))
        ywf = np.array([ 1 if pval > 0.5 else 0 for pval in predswf])

        spouse_consistency = np.mean([1 if yhs[i] == ywf[i] else 0 for i in range(len(yhs))])

        return gender_and_race_consistency, spouse_consistency
    

def get_sensitive_directions_and_projection_matrix(seed, dataset_orig_test, X_gender_train, y_gender_train, X_gender_test, y_gender_test, gender_race_features_idx = [] ):
    '''
    Description: Get the sensitive directions and projection matrix. he sensitive directions include the race and gender direction as well as the learned hyperplane that predicts gender (without using gender as a predictive feature of course).
    '''
    weights, train_logits, test_logits = SenSR.train_nn(seed, dataset_orig_test, X_gender_train, y_gender_train, X_test = X_gender_test, y_test = y_gender_test, n_units=[], l2_reg=.1, batch_size=1000, epoch=1000, verbose=True)

    n, d = weights[0].shape
    sensitive_directions = []
    full_weights = np.zeros((n+1,d))
    full_weights[0:n-1,:] = weights[0][0:n-1,:]
    full_weights[n, :] = weights[0][n-1,:]
    sensitive_directions.append(full_weights.T)
    
    if len(gender_race_features_idx) != 0:
        for idx in gender_race_features_idx:
            temp_direction = np.zeros((n+1,1)).reshape(1,-1)
            temp_direction[0, idx] = 1
            sensitive_directions.append(np.copy(temp_direction))

    sensitive_directions = np.vstack(sensitive_directions)
    tSVD = TruncatedSVD(n_components= 2 + len(gender_race_features_idx))
    tSVD.fit(sensitive_directions)
    sensitive_directions = tSVD.components_
#     print('sensitive_directions shape:', sensitive_directions.shape)

    return sensitive_directions, SenSR.compl_svd_projector(sensitive_directions)


def adult_feature_logreg(
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
def adult_proj_matrix_sensr(X_train, rind, feature_names):
    
    eg = np.zeros(X_train.shape[1])
    eg[feature_names.index('sex')] = 1.0

    er = np.zeros(X_train.shape[1])
    er[feature_names.index('race')] = 1.0

    TLR = adult_feature_logreg(X_train, rind=rind)
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
    privileged_groups = [{'sex_ Male': 1}]
    unprivileged_groups = [{'sex_ Male': 0}]

    classified_metric = ClassificationMetric(dataset_orig,
                                                     dataset_learned_model,
                                                     unprivileged_groups=unprivileged_groups,
                                                     privileged_groups=privileged_groups)

    bal_acc = compute_balanced_accuracy(classified_metric)

    gender_gap_rms, gender_max_gap = compute_gap_RMS_and_gap_max(classified_metric)
#     print("Test set: gender gap rms = %f" % gender_gap_rms)
#     print("Test set: gender max gap rms = %f" % gender_max_gap)
#     print("Test set: Balanced TPR = %f" % bal_acc)

    # wrt race
    privileged_groups = [{'race_ White': 1}]
    unprivileged_groups = [{'race_ White': 0}]

    classified_metric = ClassificationMetric(dataset_orig,
                                                     dataset_learned_model,
                                                     unprivileged_groups=unprivileged_groups,
                                                     privileged_groups=privileged_groups)

    race_gap_rms, race_max_gap = compute_gap_RMS_and_gap_max(classified_metric)
#     print("Test set: race gap rms = %f" % race_gap_rms)
#     print("Test set: race max gap rms = %f" % race_max_gap)

    return bal_acc, classified_metric.accuracy(), race_gap_rms, race_max_gap, gender_gap_rms, gender_max_gap


def run_baseline_experiment(seed, dataset_orig_test,X_train, y_train, X_test, y_test):
    return SenSR.train_nn(seed, dataset_orig_test, X_train, y_train, X_test = X_test, y_test = y_test, n_units=[100], l2_reg=0., lr = .00001, batch_size=1000, epoch=60000, verbose=True)


def run_SenSR_experiment(seed, row_list, dataset_orig_test, sind, rind, X_train, X_test, X_gender_train, X_gender_test, y_train, y_test, y_gender_train, y_gender_test, eps, fe, flr, se, slr):
    
#     sensitive_directions1, _ = get_sensitive_directions_and_projection_matrix(X_gender_train, y_gender_train, X_gender_test, y_gender_test)
#     print('sensitive_directions type:', type(sensitive_directions1))
#     print('sensitive_directions shape', sensitive_directions1.shape)
    feature_names = dataset_orig_test.feature_names
    sensitive_directions = adult_proj_matrix_sensr(X_train, rind, feature_names)
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

def run_project_experiment(seed, dataset_orig_test, X_train, X_test, X_gender_train, X_gender_test, y_train, y_test, y_gender_train, y_gender_test, directory):
    _, proj_compl = get_sensitive_directions_and_projection_matrix(seed, dataset_orig_test, X_gender_train, y_gender_train, X_gender_test, y_gender_test)

    np.save(directory+'proj_compl_'+str(seed), proj_compl)

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

    if name not in ['baseline', 'project', 'SenSR', 'adv_deb', 'proj_bst']:
        raise ValueError('You did not specify a valid experiment to run.')

    gender_race_consistencies = []
    spouse_consistencies = []

    accuracy = []
    balanced_accuracy = []

    gender_gap_rms = []
    gender_max_gap = []

    race_gap_rms = []
    race_max_gap = []
    
#     # loading projection matrices for all seeds
#     def list_files(dir):                                                                                                  
#         r = []                                                                                                            
#         files = [x for x in os.walk(dir)]                                                                                                                                              
#         return files
#     file_list = list_files('adult-seeds')
#     proj_list = file_list[0][2]
    
    
    seeds = [58185]
#     seeds = [6]
    row_list = []
    
    
    if name == 'proj_bst':
        ### Grid for Bst ###
        
        depth_grid = [3, 5, 6, 8]
        eta_grid = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]
        weight_grid = [0.1, 0.5, 1]
        lambda_grid = [1, 0.5, 0.1, 0.01]


        hypers = [depth_grid, eta_grid, weight_grid, lambda_grid]
        names = ['max_depth', 'eta', 'min_child_weight', 'lambda']

#         grid_num = len(depth_grid) *  len(eta_grid) * len(iter_grid)

        
    else:
        ### Grid for NN ###
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
        eps_grid = [epss*10]
        # eps_grid = [0.00001]
        fe_grid = [10]
        flr_grid = [flr]
        # flr_grid = [0.01*flr]
        se_grid = [10]
        slr_grid = [slr]
        # slr_grid = [0.1*slr]
        hypers = [eps_grid, fe_grid, flr_grid, se_grid, slr_grid]
        names = ['eps', 'fe', 'flr', 'se', 'slr']    
        
    
    best_param_list = []
    best_acc_list = []
    
    for seed in seeds:
        print('On experiment', seed)

        # get train/test data
        X_train, X_test, y_train, y_test, X_gender_train, X_gender_test, y_gender_train, y_gender_test, dataset_orig_train, dataset_orig_test = preprocess_adult_data(seed = seed)
        
        if name == 'proj_bst':
            num1 = y_train.sum()
            num0 = (1-y_train).sum()
            param = {
            'max_depth': 3,
            'eta': 0.01,
            'objective': 'binary:logistic',
            'min_child_weight': 1,
            'lambda': 1e-8,
            'scale_pos_weight': num0/num1,
            'nsteps':1000
            }
#             # load projection matrix
#             for ele in range(len(proj_list)):
#                 if proj_list[ele].startswith(str(seed)):
#     #                 print(proj_list[ele])
#                     proj_compl = np.load('adult-seeds/'+str(proj_list[ele]))
            

            # Get projection matrix
            _, proj_compl = get_sensitive_directions_and_projection_matrix(seed, dataset_orig_test, X_gender_train, y_gender_train, X_gender_test, y_gender_test)
            
            X_1 = X_train@proj_compl
            X_2 = X_test@proj_compl
            X_train_proj = deepcopy(X_1)
#             print('X_1 shape:', X_1.shape)
            X_train_proj = X_train_proj[:, :-2]
#             print('X_1 shape:', X_1.shape)
            X_test_proj = deepcopy(X_2)
            X_test_proj = X_test_proj[:, :-2]
#             print('X_gender_train:', X_gender_train.shape) 
            
            
            dtrain = xgb.DMatrix(data=X_train_proj, label=dataset_orig_train.labels)
            dtest = xgb.DMatrix(data=X_test_proj, label=dataset_orig_test.labels)
            watchlist = [(dtrain, 'train'), (dtest, 'test')]
        
        
        
        sind = dataset_orig_train.feature_names.index('sex_ Male')
        rind = dataset_orig_train.feature_names.index('race_ White')
        y_sex_test = X_test[:, sind]
        y_race_test = X_test[:, rind]
       
        
        count = 0
                
        for pack in product(*hypers):
            values = list(pack)
            
            if count == 0:
                value_list = [values]
            else:
                value_list.append(values)

            for ele in range(len(values)):
                param[names[ele]] = values[ele]
            print(param)
            
            if name != 'proj_bst':
                eps = param['eps']
                fe = param['fe']
                flr = param['flr']
                se = param['se']
                slr = param['slr']


            tf.compat.v1.reset_default_graph()

            # run experiments
            if name == 'baseline':
                weights, train_logits, test_logits  = run_baseline_experiment(seed, dataset_orig_test, X_train, y_train, X_test, y_test)
            elif name == 'SenSR':
                rows_list, weights, train_logits, test_logits = run_SenSR_experiment(seed, row_list, dataset_orig_test, sind, rind, X_train, X_test, X_gender_train, X_gender_test, y_train, y_test, y_gender_train, y_gender_test, eps, fe, flr, se, slr)
            elif name == 'project':
                weights, train_logits, test_logits, proj_compl = run_project_experiment(X_train, X_test, X_gender_train, X_gender_test, y_train, y_test, y_gender_train, y_gender_test)
            elif name == 'adv_deb':
                adv, preds = run_adv_deb_experiment(dataset_orig_train, dataset_orig_test)
            elif name == 'proj_bst':
                
                def balance_acc(preds, dtest):
                    ''' Balanced accuracy.'''
                    y_test = dtest.get_label()
                    y_guess = np.array([1 if pred > 0.5 else 0 for pred in preds])

                    inds0 = np.where(y_test == 0)[0]
                    inds1 = np.where(y_test == 1)[0]
                    num0 = (1-y_test).sum()
                    num1 = y_test.sum()

                    p0 = (num0 - y_guess[inds0].sum())/num0
                    p1 = y_guess[inds1].sum()/num1
                    return 'Blc', 1 - float((p0+p1)/2)
                
                eval_result = {}
                bst = xgb.train(param, dtrain, param['nsteps'], watchlist, feval=balance_acc, callbacks=[xgb.callback.record_evaluation(eval_result)])
#                 bst = xgb.train(param, dtrain, 10, watchlist, callbacks=[xgb.callback.record_evaluation(eval_result)])
                
                bl_acc_list = eval_result['test']['Blc']
#                 print('eval_result:', eval_result)
#                 print('blc:', bl_acc_list)
#                 print('blc shape:', type(bl_acc_list))
                
                best_iter = np.argmin(bl_acc_list)
#                 print(best_iter)
#                 print('value_list:', value_list)
                
                best_param = value_list[count].copy()
#                 print(value_list[0])
#                 print(best_param)
                best_param.append(best_iter)
#                 print(best_param)
#                 print(best_param_list)
                best_param_list.append(best_param)
#                 print(best_param_list)
                best_acc_list.append(bl_acc_list[best_iter])
                count += 1
                best_param = []
                
            # get race/gender and spouse consistency

            if name == 'project':
                gender_race_consistency = get_consistency(X_test, weights = weights, proj = proj_compl)
            elif name == 'adv_deb':
                gender_race_consistency = get_consistency(X_test, adv = adv, dataset_orig_test = dataset_orig_test)
            elif name == 'proj_bst':
                # undo
                pass
#                 gender_race_consistency, spouse_consistency = get_consistency_bst(X_test, bst=bst, adv = 0, dataset_orig_test = dataset_orig_test)
            else:
                gender_race_consistency, spouse_consistency = get_consistency(X_test, weights = weights)
            if name != 'proj_bst':
                print('gender_race_consistency', gender_race_consistency)
                print('spouse consistency', spouse_consistency)

            # get accuracy, balanced accuracy, gender/race gap rms, gender/race max gap

    #         if name != 'adv_deb':
    #             np.save(directory+'weights_'+str(i), weights)
            if name != 'proj_bst':

                preds = np.argmax(test_logits, axis = 1)
            else:
                predst = bst.predict(dtest)
                preds = np.array([1 if pval > 0.5 else 0 for pval in predst])
                
#             if name != 'proj_bst':
            bal_acc_temp, acc_temp, race_gap_rms_temp, race_max_gap_temp, gender_gap_rms_temp, gender_max_gap_temp = get_metrics(dataset_orig_test, preds)
#             print('Blc:', bal_acc_temp)
#             print('acc:', acc_temp)

            yt = dataset_orig_test.labels
            yt = np.reshape(yt, (-1, ))
            y_guess = preds
#             print('Acc:', accuracy)
#             print("SEX")
            base_sex = script.group_metrics(yt, y_guess, y_sex_test, label_good=1)
#             print("RACE")
            base_race = script.group_metrics(yt, y_guess, y_race_test, label_good=1)
            
#             print('Saving results ......')
#             save_data = {}
#             save_data['param'] = 'No parameter selection'
#             save_data['seed'] =  seed
#             save_data['acc'] = base_sex[-1]
#             save_data['bl_acc'] = bal_acc_temp
#             save_data['grcons'] = gender_race_consistency
#             save_data['scons'] = spouse_consistency
#             save_data['RMS(G)'] = base_sex[4]
#             save_data['MAX(G)'] = base_sex[5]
#             save_data['AOD(G)'] = base_sex[6]
#             save_data['EOD(G)'] = base_sex[7]
#             save_data['SPD(G)'] = base_sex[8]
#             save_data['RMS(R)'] = base_race[4]
#             save_data['MAX(R)'] = base_race[5]
#             save_data['AOD(R)'] = base_race[6]
#             save_data['EOD(R)'] = base_race[7]
#             save_data['SPD(R)'] = base_race[8]
#             row_list.append(save_data)

#     dfsave_data = pd.DataFrame(data=row_list)
#     dfsave_data.to_csv('ave.csv', index=True)
#     best_index = np.argmin(best_acc_list)
#     best_param_final = best_param_list[best_index]
#     print('best param:', best_param_final)
#     print('best acc:', best_acc_list[best_index])
    print('DONE')
    
    return 0

run_experiments('proj_bst')
