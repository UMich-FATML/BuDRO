import scipy as sp
import xgboost as xgb
import os
import numpy as np
import tensorflow as tf
import sys
import time
import pandas as pd
from copy import deepcopy
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from aif360.algorithms.preprocessing.reweighing import Reweighing
from sklearn.decomposition import TruncatedSVD
from IPython.display import display, Markdown, Latex
from sklearn.linear_model import LogisticRegressionCV
import data_pre
import script


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


def preprocess_compas_data(name, seed=0):
    '''
    Description: Ths code (1) standardizes the continuous features, (2) one hot encodes the categorical features, (3) splits into a train (60% for training set, 20% for validation set) and test set (20%), (4) based on this data, create another copy where race is deleted as a predictive feature and the feature we predict is race (learning the sensitive directions)
    Input: seed: the seed used to split data into train/test
    '''
    # Get the dataset 
    dataset_orig = get_compas_orig()
    # we will standardize continous features
    continous_features = [
            'priors_count'
        ]
    continous_features_indices = [
            dataset_orig.feature_names.index(feat) 
            for feat in continous_features
        ]
    
    if name != 'rw_bst':
        # Get the dataset and split into train, valid and test
        dataset_orig_train, dataset_orig_vt = dataset_orig.split([0.8-0.2], shuffle=True)
        dataset_orig_valid, dataset_orig_test = dataset_orig_vt.split([0.5], shuffle=True)

        SS = StandardScaler().fit(dataset_orig_train.features[:, continous_features_indices])
        dataset_orig_train.features[:, continous_features_indices] = SS.transform(dataset_orig_train.features[:, continous_features_indices])
        dataset_orig_test.features[:, continous_features_indices] = SS.transform(dataset_orig_test.features[:, continous_features_indices])
        dataset_orig_valid.features[:, continous_features_indices] = SS.transform(dataset_orig_valid.features[:, continous_features_indices])

        X_train = dataset_orig_train.features
        X_test = dataset_orig_test.features
        X_valid = dataset_orig_valid.features
        y_train = dataset_orig_train.labels
        y_test = dataset_orig_test.labels
        y_valid = dataset_orig_valid.labels
    else:
        # Get the dataset and split into train, valid and test
        dataset_orig_train, dataset_orig_test = dataset_orig.split([0.8], shuffle=True)

        SS = StandardScaler().fit(dataset_orig_train.features[:, continous_features_indices])
        dataset_orig_train.features[:, continous_features_indices] = SS.transform(dataset_orig_train.features[:, continous_features_indices])
        dataset_orig_test.features[:, continous_features_indices] = SS.transform(dataset_orig_test.features[:, continous_features_indices])

        X_train = dataset_orig_train.features
        X_test = dataset_orig_test.features
        y_train = dataset_orig_train.labels
        y_test = dataset_orig_test.labels

    return X_train, X_test, y_train, y_test, dataset_orig_train, dataset_orig_test


def compas_feature_logreg(
        X_train,
        rind=None,
        feature_names=None,
        test=True,
        labels=None
        ):
    
    X_test=None
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
    
    if test and X_test is not None:
        outputs = LR.predict(np.vstack((X_train,X_test)))
        print("Training error of LR classifier: {}".format(
                np.abs(outputs-targets).sum()/XLR.shape[0]
            ))

    return LR


def compas_proj_matrix(X_train, rind, feature_names, test=False,save=False):
    '''
    Description: Get the sensitive directions and projection matrix. The sensitive directions include the race and gender direction as well as the learned hyperplane that predicts race (without using race as a predictive feature of course).
    '''
    eg = np.zeros(X_train.shape[1])
    eg[feature_names.index('sex')] = 1.0

    er = np.zeros(X_train.shape[1])
    er[feature_names.index('race')] = 1.0

    TLR = compas_feature_logreg(X_train, rind=rind, feature_names=feature_names, test=test)
    wg = TLR.coef_[0]

    A = np.array([wg, eg, er]).T

    return script.proj_matrix_gen(X_train, A, test, save)

def race_and_sex_cons(bst, X_test, sind, rind, name):
    
    
    # Calculate gender consistency
    Xg = deepcopy(X_test)

    Xg[:, sind] = np.zeros(Xg.shape[0])
    preds0 = bst.predict(xgb.DMatrix(Xg))
    y0 = np.array([ 1 if pval > 0.5 else 0 for pval in preds0])

    Xg[:, sind] = np.ones(Xg.shape[0])
    preds1 = bst.predict(xgb.DMatrix(Xg))
    y1 = np.array([ 1 if pval > 0.5 else 0 for pval in preds1])

    gender_consistency = (y0*y1).sum() + ((1-y0)*(1-y1)).sum()
        
    if name != 'proj_bst':
        # Calculate race consistency
        Xr = deepcopy(X_test)

        Xr[:, rind] = np.zeros(Xr.shape[0])
        preds0 = bst.predict(xgb.DMatrix(Xr))
        y0 = np.array([ 1 if pval > 0.5 else 0 for pval in preds0])

        Xr[:, rind] = np.ones(Xr.shape[0])
        preds1 = bst.predict(xgb.DMatrix(Xr))
        y1 = np.array([ 1 if pval > 0.5 else 0 for pval in preds1])
        
        race_consistency = (y0*y1).sum() + ((1-y0)*(1-y1)).sum()
    else:
        race_consistency = X_test.shape[0]
        
    return gender_consistency, race_consistency


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
    # An example for parameters (feed in the optimal hyper-parameters):    
    if name == 'proj_bst':
        max_depth = 4
        eta = 7.5e-4
        niter = 2000
    elif name =='bst':
        max_depth = 3
        eta = 5e-4
        niter = 1600
    else:
        niter = 100

    for seed in seeds:
        print('On experiment', seed)

        # get train/test data
        np.random.seed(seed)
        X_train, X_test, y_train, y_test, dataset_orig_train, dataset_orig_test = preprocess_compas_data(name, seed = seed)
        
        # Get binary gender and race features
        feature_names = dataset_orig_train.feature_names
        sind = feature_names.index('sex')
        rind = feature_names.index('race')
        y_sex_test = X_test[:, sind]
        y_race_test = X_test[:, rind]
        
        # Calculte '# 1' and # 0' of labels in training set
        num1 = dataset_orig_train.labels.sum()
        num0 = (1-dataset_orig_train.labels).sum()
        
        # Set parameters for XGBoost
        if name != 'rw_bst':
            param = {
            'max_depth': max_depth,
            'eta': eta,
            'objective': 'binary:logistic',
            'min_child_weight': 0.1/y_train.shape[0],
            'lambda': 1e-8,
            'scale_pos_weight': 1,
            'nsteps':niter
            }
        else:
            param = {
            'objective': 'binary:logistic',
            'min_child_weight': 1,
            'nsteps':niter
            }
        
        # Prepare XGBoost inputs
        if name == 'proj_bst':
            # Get projection matrix
            proj_compl = compas_proj_matrix(X_train, rind, feature_names, test=False,save=False)
            
            X_1 = X_train@proj_compl
            X_2 = X_test@proj_compl
            # Remove proctected features from training data
            X_train_proj = deepcopy(X_1)
            X_train_proj = np.delete(X_train_proj, 1, 1)
            X_test_proj = deepcopy(X_2)
            X_test_proj = np.delete(X_test_proj, 1, 1)
            
            dtrain = xgb.DMatrix(data=X_train_proj, label=dataset_orig_train.labels)
            dtest = xgb.DMatrix(data=X_test_proj, label=dataset_orig_test.labels)
            
        elif name == 'rw_bst':
            # Set privileged_groups and unprivileged_groups for reweighing methods
            privileged_groups = [{'race': 1}]
            unprivileged_groups = [{'race': 0}]
            
            RW = Reweighing(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
            RW.fit(dataset_orig_train)
            dataset_transf_train = RW.transform(dataset_orig_train)
            X_train = dataset_transf_train.features
            y_train = dataset_transf_train.labels
            w_train = dataset_transf_train.instance_weights.ravel()
            dtrain = xgb.DMatrix(data=X_train, label=y_train, weight=w_train)
            dtest = xgb.DMatrix(data=X_test, label=dataset_orig_test.labels)
            
        else:
            dtrain = xgb.DMatrix(data=X_train, label=dataset_orig_train.labels)
            dtest = xgb.DMatrix(data=X_test, label=dataset_orig_test.labels)
            
        watchlist = [(dtrain, 'train'), (dtest, 'test')]                      
        
        # Run XGBoost
        bst = xgb.train(param, dtrain, param['nsteps'], watchlist)

        ###Calculating all evaluation metrics###
        print('Calculating evaluation metrics ......')
        #     y_train = np.reshape(y_train, (-1, ))
        #     y_test = np.reshape(y_test, (-1, ))
        
        # get race/gender and spouse consistency
        if name == 'proj_bst':
            X_t = deepcopy(X_test)
            X_t = np.delete(X_t, 1, 1)
            gender_consistency, race_consistency = race_and_sex_cons(bst, X_t, sind, rind, name)
        else: 
            gender_consistency, race_consistency = race_and_sex_cons(bst, X_test, sind, rind, name)
        
        # Get binary predict lables from XGBoost
        predst = bst.predict(dtest)
        preds = np.array([1 if pval > 0.5 else 0 for pval in predst])
        
        # Calculate other evaluation metrics
        yt = dataset_orig_test.labels
        y_guess = preds
        # balanced accuracy
        p0, p1, _ = script.balanced_accuracy(bst, dtest, yt)
        yt = np.reshape(yt, (-1, ))
        base_sex = script.group_metrics(yt, y_guess, y_sex_test, label_good=1)
        base_race = script.group_metrics(yt, y_guess, y_race_test, label_good=1)

        print('Saving results ......')
        save_data = {}
        if name != 'rw_bst':
            save_data['max_depth'] = param['max_depth']
            save_data['eta'] = param['eta']
            save_data['min_child_weight'] = param['min_child_weight']
            save_data['lambda'] = param['lambda']
        save_data['seed'] =  seed
        save_data['acc'] = base_sex[-1]
        save_data['bl_acc'] = (p0+p1)/2
        save_data['gcons'] = gender_consistency / X_test.shape[0]
        save_data['rcons'] = race_consistency / X_test.shape[0]
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
    filename = 'result:%d' % seed + '_' + name + '_Compas.csv'
    dfsave_data.to_csv(filename, index=True)

    print('DONE')
    return 0

# Run experiments
seed = [1, 6, 8, 26, 27, 30, 36, 47, 63, 70]
run_experiments('proj_bst', seed) # for projecting baseline with random seed 
run_experiments('rw_bst', seed) # for reweighing baseline with random seed 
run_experiments('bst', seed) # for vanilla baseline with random seed 
