import numpy as np
import pandas as pd
import os
from copy import deepcopy

from sklearn import preprocessing
from aif360.datasets import BinaryLabelDataset
import tensorflow as tf
import time
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy.stats import sem
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.algorithms.preprocessing.reweighing import Reweighing
from sklearn.decomposition import TruncatedSVD
from IPython.display import display, Markdown, Latex
from sklearn.linear_model import LogisticRegressionCV

import SenSR
import script
from itertools import product

import xgboost as xgb


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


def compute_balanced_accuracy(data_set):
    '''
    Description: computes the balanced accuracy, i.e. the average of TPR and TNR
    Input: data_set: a BinaryLabelDataset (from the aif360 module)
    '''
    TPR = data_set.true_positive_rate()
    TNR = data_set.true_negative_rate()
    return 0.5*(TPR+TNR)

def get_consistency_bst(name, X, bst=0, proj = 0, gender_idx = 39, race_idx = 40, relationship_idx = [33, 34, 35, 36, 37, 38], husband_idx = 33, wife_idx = 38, dataset_orig_test = 0):
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
    if name != 'proj_bst':
        gender_race_idx = [gender_idx, race_idx]

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
    else:
        gender_and_race_consistency = 1

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

def get_sensitive_directions_and_projection_matrix(X_gender_train, y_gender_train, X_gender_test, y_gender_test, gender_race_features_idx = [0, 1] ):
    '''
    Description: Get the sensitive directions and projection matrix. The sensitive directions include the race and gender direction as well as the learned hyperplane that predicts gender (without using gender as a predictive feature of course).
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
    Description: This code computes balanced accuracy
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

    return bal_acc


def run_experiments(name, seed):
    '''
    Description: Scripts for running baselines based on boosting.
    Run each experiment length(seed) times where a new train/test split is generated. 

    Inputs: 
    name: name of the experiment. Valid choices are baseline (boosting), project and reweighing
    seed: train/test splitting seed
    
    Outputs:
    Saving all evaluation metrics associated with hyper-parameters to a csv file
    '''

    if name not in ['rw_bst', 'proj_bst', 'bst']:
        raise ValueError('You did not specify a valid experiment to run.')
    
    # Loading train/test splitting seeds
    seeds = seed

    # List for saving evaluation metrics
    row_list = []
    
    ### Parameters for Bst ###
    # An example for parameters:    
    max_depth = 3
    eta = 0.2
    min_child_weight = 0.5
    bst_lambda = 0.1
    niter = 100
    
    for seed in seeds:
        print('On experiment', seed)

        # get train/test data
        X_train, X_test, y_train, y_test, X_gender_train, X_gender_test, y_gender_train, y_gender_test, dataset_orig_train, dataset_orig_test = preprocess_adult_data(seed = seed)
        
        # Get binary gender and race features
        sind = dataset_orig_train.feature_names.index('sex_ Male')
        rind = dataset_orig_train.feature_names.index('race_ White')
        y_sex_test = X_test[:, sind]
        y_race_test = X_test[:, rind]
        
        # Calculte '# 1' and # 0' of labels in training set
        num1 = dataset_orig_train.labels.sum()
        num0 = (1-dataset_orig_train.labels).sum()
        
        # Set parameters for XGBoost
        param = {
        'max_depth': max_depth,
        'eta': eta,
        'objective': 'binary:logistic',
        'min_child_weight': min_child_weight,
        'lambda': bst_lambda,
        'scale_pos_weight': num0/num1,
        'nsteps':niter
        }
        
        # Prepare XGBoost inputs
        if name == 'proj_bst':
            # Get projection matrix
            _, proj_compl = get_sensitive_directions_and_projection_matrix(X_gender_train, y_gender_train, X_gender_test, y_gender_test)
            
            X_1 = X_train@proj_compl
            X_2 = X_test@proj_compl
            # Remove proctected features from training data
            X_train_proj = deepcopy(X_1)
            X_train_proj = X_train_proj[:, :-2]
            X_test_proj = deepcopy(X_2)
            X_test_proj = X_test_proj[:, :-2]
            
            dtrain = xgb.DMatrix(data=X_train_proj, label=dataset_orig_train.labels)
            dtest = xgb.DMatrix(data=X_test_proj, label=dataset_orig_test.labels)
            
        elif name == 'rw_bst':
            # Set privileged_groups and unprivileged_groups for reweighing methods
            privileged_groups = [{'sex_ Male': 1}]
            unprivileged_groups = [{'sex_ Male': 0}]
            
            RW = Reweighing(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
            RW.fit(dataset_orig_train)
            dataset_transf_train = RW.transform(dataset_orig_train)
            X_train = dataset_transf_train.features
            y_train = dataset_transf_train.labels
            w_train = dataset_transf_train.instance_weights.ravel()
            dtrain = xgb.DMatrix(data=X_train, label=dataset_orig_train.labels, weight=w_train)
            dtest = xgb.DMatrix(data=X_test, label=dataset_orig_test.labels)
            
        else:
            dtrain = xgb.DMatrix(data=X_train, label=dataset_orig_train.labels)
            dtest = xgb.DMatrix(data=X_test, label=dataset_orig_test.labels)
            
        watchlist = [(dtrain, 'train'), (dtest, 'test')]                      
                
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

        # Run XGBoost
        bst = xgb.train(param, dtrain, param['nsteps'], watchlist, feval=balance_acc)

        ###Calculating all evaluation metrics###
        print('Calculating evaluation metrics ......')
        
        # get race/gender and spouse consistency
        if name == 'proj_bst':
            X_t = deepcopy(X_test)
            X_t = X_t[:, :-2]
            gender_race_consistency, spouse_consistency = get_consistency_bst(name, X_t, bst=bst, dataset_orig_test = dataset_orig_test)
        else: 
            gender_race_consistency, spouse_consistency = get_consistency_bst(name, X_test, bst=bst, dataset_orig_test = dataset_orig_test)
        
        # Get binary predict lables from XGBoost
        predst = bst.predict(dtest)
        preds = np.array([1 if pval > 0.5 else 0 for pval in predst])
        
        # balanced accuracy
        bal_acc_temp = get_metrics(dataset_orig_test, preds)
        
        # Calculate other evaluation metrics
        yt = dataset_orig_test.labels
        yt = np.reshape(yt, (-1, ))
        y_guess = preds
        base_sex = script.group_metrics(yt, y_guess, y_sex_test, label_good=1)
        base_race = script.group_metrics(yt, y_guess, y_race_test, label_good=1)

        print('Saving results ......')
        save_data = {}
        save_data['max_depth'] = param['max_depth']
        save_data['eta'] = param['eta']
        save_data['min_child_weight'] = param['min_child_weight']
        save_data['lambda'] = param['lambda']
        save_data['seed'] =  seed
        save_data['acc'] = base_sex[-1]
        save_data['bl_acc'] = bal_acc_temp
        save_data['grcons'] = gender_race_consistency
        save_data['scons'] = spouse_consistency
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
    filename = 'result:%d' % seed + '_' + name + '_Adult.csv'
    dfsave_data.to_csv(filename, index=True)

    print('DONE')
    return 0

# Run experiments
seed = [1]
run_experiments('proj_bst', seed) # for projecting baseline with random seed 1
