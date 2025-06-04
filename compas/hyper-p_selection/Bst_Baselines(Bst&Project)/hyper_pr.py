#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.model_selection import StratifiedShuffleSplit

from copy import deepcopy
import scipy as sp

import xgboost as xgb

from sklearn.linear_model import LogisticRegressionCV

from optparse import OptionParser
from collections import OrderedDict

import os
import numpy as np
import tensorflow as tf
import sys

import time

import compas_proc
import script
# import tf_xgboost_log as txl

from itertools import product


# Data processing
_dtype = tf.float32
seeds = [1]
reweight = False
flag_project = True

depth_grid = [2, 3, 4, 5, 6]
eta_grid = [0.00025, 0.0005, 0.00075, 0.001]
iter_grid = [500, 800, 1000, 1100, 1200, 1300, 1500, 1600, 1800, 2000]


hypers = [depth_grid, eta_grid, iter_grid]
names = ['max_depth', 'eta', 'nsteps']

grid_num = len(depth_grid) *  len(eta_grid) * len(iter_grid)

ave_acc = np.zeros(grid_num) 
bl_acc = np.zeros(grid_num) 

for seed in seeds:
    print('seed:%d' %seed)
    count = 0
    np.random.seed(seed)
    X_train, X_test, y_train, y_test, y_sex_train, y_sex_test, y_race_train, y_race_test, Corig, proj, w_train, w_test, features, X_valid, y_valid, w_valid = adult_proc.adult_setup(seed=seed, pct=0.8)

    dtrain, dtest, dvalid, watchlist, num0, num1 = adult_proc.adult_prep_baseline_xgb(X_train, X_test, X_valid, y_train, y_test, y_valid, w_train, w_test, w_valid, proj, reweight, flag_project)
    # initialize parameters
    param = {
        'max_depth': 3,
        'eta': 0.01,
        'objective': 'binary:logistic',
        'min_child_weight': 0.1/y_train.shape[0],
        'lambda': 1e-8,
        'scale_pos_weight': 1,
        'nsteps':100
    }
    
    for pack in product(*hypers):
        values = list(pack)
        
        if seed==1:
            if count == 0:
                value_list = [values]
            else:
                value_list.append(values)
                
        for ele in range(len(values)):
            param[names[ele]] = values[ele]
        print(param)
        
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

        bst = xgb.train(param, dtrain, param['nsteps'], watchlist, feval=balance_acc)
        y_true = y_test
        predst = bst.predict(dtest)
        y_guess = np.array([1 if pval > 0.5 else 0 for pval in predst])
        p0, p1, _ = script.balanced_accuracy(bst, dvalid, y_valid)
        bl_acc[count] += (p0+p1)/2

        ave_acc[count] += sum(y_true[_] == y_guess[_] for _ in range(len(y_true))) / len(y_guess)
        count +=1
        
best_value = value_list[np.argmax(bl_acc)]
print('best value:', best_value)
