from sklearn.preprocessing import StandardScaler
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
import pandas as pd
import tensorflow as tf
import sys

import time

import compas_proc
import script
import tf_xgboost_log as txl
from itertools import product


def parse_args():
    
    parser = OptionParser()

    parser.add_option("--sgd", action="store_true", dest="sgd", default=False)
    parser.add_option("--dual", action="store_true", dest="dual", default=False)

    parser.add_option("--eps", type="float", dest="eps")
    parser.add_option("--gamma", type="float", dest="gamma_reg")

    parser.add_option("--lambda", type="float", dest="lamb")
    parser.add_option("--max_depth", type="int", dest="max_depth")
    parser.add_option("--eta_lr", type="float", dest="eta_lr")
    parser.add_option("--min_child_weight", type="float", dest="min_child_weight")
    parser.add_option("--scale_pos_weight", type="float", dest="scale_pos_weight")
    parser.add_option("--n_iter", type="int", dest="n_iter")
    #parser.add_option(, type="float", dest=)
 
    parser.add_option("--sgd_init", type="float", dest="sgd_init")
    parser.add_option("--epoch", type="int", dest="epoch")
    parser.add_option("--batch_size", type="int", dest="batch_size")
    parser.add_option("--lr", type="float", dest="lr")
    parser.add_option("--momentum", type="float", dest="momentum")
    
    # seed for train/test split
    parser.add_option("--seed", type="int", dest="seed")
    
    parser.add_option("-f", "--file", dest="filename",
                  help="write report to FILE", metavar="FILE")
    
    (options, args) = parser.parse_args()
 
    return options


def main():

    _baseline = True
    _dtype = tf.float32

    options = parse_args()
    print(options)

    # method parameters
    use_sgd = options.sgd
    use_dual = options.dual

    eps = options.eps

    # xgboost parameters
    param = dict()
    param['objective'] = 'binary:logistic'
    param['max_depth'] = options.max_depth
    param['lambda'] = options.lamb
    param['eta'] = options.eta_lr
    param['min_child_weight'] = options.min_child_weight
    param['scale_pos_weight'] = options.scale_pos_weight
    n_iter = options.n_iter

    names = ['eps', 'depth', 'eta', 'weight', 'lamb', 'pos']
    values = [ eps,
        param['max_depth'],
        param['eta'],
        param['min_child_weight'],
        param['lambda'],
        param['scale_pos_weight'],
        ]

    reweight=False
    flag_project = False
    
    
    #     Build hyper-parameter grid
    depth_grid = [2, 3, 4, 5, 6, 8]
    eta_grid = [0.00001, 0.000015, 0.00002, 0.00005, 0.0001]
    niter_grid = [200]
    eps_grid = [0.01, 0.05, 0.075, 0.1, 0.12, 0.15, 0.2, 0.5]


#     eps_grid = [0.01, 0.05]

    hypers = [depth_grid, eta_grid, niter_grid, eps_grid]

    names = ['max_depth', 'eta', 'nsteps']
    seeds = [6]
    rows_list = []

    
    # TRAIN WITH THESE PARAMETERS
    print("LOADING COMPAS DATA", flush=True)
    for seed in seeds:
        print('seed:%d' %seed)
        np.random.seed(seed)
        X_train, X_test, y_train, y_test, y_sex_train, y_sex_test, y_race_train, y_race_test, feature_names, Corig, dtrain, dtest, watchlist, param, bst, proj, w_train, w_test = compas_proc.compas_setup(reweight, flag_project, nsteps=200, baseline=True, seed=seed,param=None, pct=0.8, loadData=False)

        sind = feature_names.index('sex')
        rind = feature_names.index('race')
        print('X_train:', X_train.shape)

        # Refine the labels
        y = y_train
        yt = y_test

        inds0 = np.where(y==0)[0]
        inds1 = np.where(y==1)[0]

        C = tf.constant(Corig, _dtype)

        for pack in product(*hypers):

            values = list(pack)
            #  setting values to params
            for ele in range(len(names)):
                param[names[ele]] = values[ele]
            eps = values[-1]
#            gamma_reg = values[-1]
            
            print('parameter:', param)
            ## FAIR TRAINING
            t_s = time.time()
            rows_list, fst, pi =  txl.boost_sinkhorn_tf(
                    seed,
                    sind,
                    rind,
                    y_sex_test,
                    y_race_test,
                    X_train,
                    y,
                    C, 
                    param['nsteps']+1, 
                    rows_list,
                    X_test=X_test,
                    y_test=yt,
        #            pred=fst.predict(dtrain, output_margin=False), 
                    pred = None,
                    eps=eps, 
                    gamma_reg = 0.00005,
                    param=param,
                    bst=None,
                    lowg=-10.0,
                    highg=10.0,
                    verbose=False,
                    verify=True,
                    roottol=10**(-16),
                    dtype=_dtype,
                    max_float_pow = 65.,
                    outfunc=None
                    )
            print('Took %f' % (time.time() - t_s))

            # output some summary imformation
            print("BOOSTED CLASSIFIER FINAL INFORMATION")
            p0, p1, _ = script.balanced_accuracy(fst, dtest, yt)
            print("p0: {}; p1: {}; balanced: {}".format(p0, p1, (p0+p1)/2))

            predst = fst.predict(dtest)
            gcons = compas_proc.sex_cons(fst, X_test, sind, rind) / X_test.shape[0]
            print('gender_cons:', gcons)
            y_guess = np.array([1 if pval > 0.5 else 0 for pval in predst])
            print("SEX")
            base_sex = script.group_metrics(yt, y_guess, y_sex_test, label_good=1)
            print("RACE")
            base_race = script.group_metrics(yt, y_guess, y_race_test, label_good=1)
            
    dfsave_data = pd.DataFrame(data=rows_list)
    dfsave_data.to_csv('ave.csv', index=True)
    print('DONE')
    
    return 0

main()
