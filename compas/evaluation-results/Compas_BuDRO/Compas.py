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
import tf_xgboost_log as txl
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


def preprocess_compas_data(seed=0):
    '''
    Description: Ths code (1) standardizes the continuous features, (2) one hot encodes the categorical features, (3) splits into a train (80%) and test set (20%), (4) based on this data, create another copy where race is deleted as a predictive feature and the feature we predict is race (learning the sensitive directions)
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
    name: name of the experiment. Valid choice is BuDRO
    seed: train/test splitting seed

    Outputs:
    Saving all evaluation metrics associated with hyper-parameters to a csv file
    '''
    if name not in ['BuDRO']:
        raise ValueError('You did not specify a valid experiment to run.')
    
    _dtype = tf.float32
    
    # Loading train/test splitting seeds
    seeds = seed

    # List for saving evaluation metrics
    row_list = []

    ### Parameters for Bst ###
    # An example for parameters (feed in the optimal hyper-parameters):    
    max_depth = 2
    eta = 0.000015
    bst_lambda = 1e-8
    niter = 68
    ### Parameters for BuDRO (if 'BuDRO') ###
    if name == 'BuDRO':
        eps = 0.12

    for seed in seeds:
        print('On experiment', seed)

        # get train/test data
        np.random.seed(seed)
        X_train, X_test, y_train, y_test, dataset_orig_train, dataset_orig_test = preprocess_compas_data(seed = seed)
        
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
        param = {
        'max_depth': 3,
        'eta': 0.01,
        'objective': 'binary:logistic',
        'min_child_weight': 0.1/y_train.shape[0],
        'lambda': bst_lambda,
        'scale_pos_weight': num0/num1,
        'nsteps':niter
        }
        
        # Prepare XGBoost inputs
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
        bst = xgb.train(param, dtrain, 200, watchlist, feval=balance_acc)
        
        if name == 'BuDRO':
            param['max_depth'] = max_depth
            param['eta'] = eta
            y_train = np.reshape(y_train, (-1, ))
            y_test = np.reshape(y_test, (-1, ))
            # Get projection matrix for calculating the 'similarity distance'
            proj = compas_proj_matrix(X_train, rind, feature_names, test=False,save=False)
            projData = np.matmul(X_train, proj)
            Corig = sp.spatial.distance.squareform(sp.spatial.distance.pdist(projData, metric='sqeuclidean'))
            C = tf.constant(Corig, _dtype)
            print('Corig:', Corig.shape)
            y = y_train
            yt = y_test
            
            # Run BuDRO
            fst, pi =  txl.boost_sinkhorn_tf(seed, sind, rind, y_sex_test, y_race_test, X_train, y, C, param['nsteps']+1, X_test=X_test,
                    y_test=yt, pred = None, eps=eps, gamma_reg = 0.00005, param=param, bst=None, lowg=-10.0,
                    highg=10.0, verbose=False, verify=True, roottol=10**(-16), dtype=_dtype, max_float_pow = 65.)

        ###Calculating all evaluation metrics###
        print('Calculating evaluation metrics ......')
        
        # get race/gender and spouse consistency
        if name == 'BuDRO':
            gender_consistency, race_consistency = race_and_sex_cons(fst, X_test, sind, rind, name)
        else: 
            gender_consistency, race_consistency = race_and_sex_cons(bst, X_test, sind, rind, name)
        
        # Get binary predict lables from XGBoost
        if name == 'BuDRO':
            predst = fst.predict(dtest)
        else:
            predst = bst.predict(dtest)
        preds = np.array([1 if pval > 0.5 else 0 for pval in predst])
        
        # Calculate other evaluation metrics
        yt = dataset_orig_test.labels
        y_guess = preds
        # balanced accuracy
        if name == 'BuDRO':
            p0, p1, _ = script.balanced_accuracy(fst, dtest, yt)
        else:
            p0, p1, _ = script.balanced_accuracy(bst, dtest, yt)
            
        yt = np.reshape(yt, (-1, ))
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
run_experiments('BuDRO', seed) # for BuDRO with random seed 1
