import sys
from optparse import OptionParser
from itertools import product
import time

import os
import shutil

import numpy as np
import scipy as sp
import tensorflow as tf
import xgboost as xgb

import german_proc
import script
import tf_xgboost_log as txl

_dtype = tf.float32
_verbose = False

# An example of running the fair method, potentially using multiple
# hyperparameter values

# path with the actual boosting scripts in it
sys.path.append('../scripts')

# call this as a command
def parse_args():
    
    parser = OptionParser()

    # seed for train/test split
    parser.add_option("--seed", type="int", dest="seed")

    # value of epsilon
    parser.add_option("--eps", type="float", dest="eps")
    parser.add_option("--max_depth", type="int", dest="max_depth")
    parser.add_option("--eta_lr", type="float", dest="eta_lr")

    parser.add_option("--dual", action="store_true", dest="dual", default=False)

    parser.add_option(
            "--include_age_bin", action="store_true", dest="agebin", default=False
            )
    (options, args) = parser.parse_args()
 
    return options

# balanced accuracy
def balacc(predt: np.ndarray, dtrain: xgb.DMatrix):
    y_true = dtrain.get_label()
    y_guess = np.array([1 if pval >= 0.5 else 0 for pval in predt])

    yt0 = np.where(y_true==0)[0]
    yt1 = np.where(y_true==1)[0]

    p0 = (yt0.shape[0] - y_guess[yt0].sum())/yt0.shape[0]
    p1 = y_guess[yt1].sum()/yt1.shape[0]

    return 'balacc', (p0 + p1)/2

def main(results_path = 'YOUR_RESULTS_PATH_HERE'):

    options = parse_args()
    print(options)
    seed = options.seed
    eps = options.eps

    use_dual = options.dual
    agebin = options.agebin

    depth = options.max_depth
    eta = options.eta_lr

    #### SET UP THE DATA
    # Load the correct data
    # basically follow german_setup_newmetric in case I want the projection
    # matrix for anything
    orig_dataset = german_proc.get_german_data()
    X_train, X_test, y_train, y_test, y_age_train, y_age_test, feature_names, traininds, testinds = german_proc.get_german_train_test_age(orig_dataset, pct=0.8, removeProt=False, seed=seed)

    aind = feature_names.index('age')
    abind = feature_names.index('age_bin')

    # Get status_inds and binary gender labels
    status_inds = []
    y_sex_bin = np.zeros(X_train.shape[0])
    yt_sex = np.zeros(X_test.shape[0])

    for i, nom in enumerate(feature_names):
        if "personal_status" in nom:
            status_inds.append(i)

            # Males are 1, females are 0 in the following
            if 'A92' not in nom and 'A95' not in nom:
                y_sex_bin += X_train[:,i]
                yt_sex += X_test[:,i]

    print("Personal status indices: {}".format(status_inds))

    muppet = y_sex_bin.max()
    if muppet > 1: 
        print("Warning: in train set, you messed up the gender binary")
    muppet = yt_sex.max()
    if muppet > 1: 
        print("Warning: in test set, you messed up the gender binary")

    # Get binary age labels and project out if needed
    y_age_bin = np.copy(X_train[:,abind])
    yt_age = np.copy(X_test[:, abind])

    if not agebin:
        print("Eliminating binary age attribute")
        X_train[:,abind] = 0
        X_test[:, abind] = 0

    # project the training data and compute pairwise distances.
    # Make the required tensorflow constant array C
    RCV = german_proc.train_ridge(X_train, y_age_train, aind=aind, abind=None)
    proj = german_proc.german_proj_mat(RCV, aind)
    projData = np.matmul(X_train, proj)

    print("Calculate pairwise distances... ", end=" ", flush=True)
    Corig = sp.spatial.distance.squareform(
            sp.spatial.distance.pdist(
                projData,
                metric='sqeuclidean'
            ))
    print("done.", flush=True)
    C = tf.constant(Corig, dtype=_dtype)

    n = Corig.shape[0]

    dtrain, dtest, watchlist, param, _, _ = german_proc.prep_baseline_xgb(X_train, X_test, y_train, y_test)


    # xgboost parameter grid
    lambda_grid = [0.001, 0.1, 1, 10, 100, 1000] #6
    weight_grid = [0.01, 0.1, 1, 10] #4

    depth_grid = [4,7,10,13] #4
    eta_grid = [0.0001, 0.001, 0.005, 0.01, 0.05 ] #5

    pos_grid = [0.0]  #1
    n_iter = 500 

    # test the script
    #lambda_grid = [ 10 ]
    #depth_grid = [10]
    #eta_grid = [0.05] 
    #weight_grid = [ 2] 
    #pos_grid = [0.0]
    #n_iter = 1000
    
    #hypers = [depth_grid, eta_grid, weight_grid, lambda_grid, pos_grid]
    hypers = [weight_grid, lambda_grid, pos_grid]
    names = ['depth', 'eta', 'weight', 'lamb', 'pos']

    names += ['seed']
    names = ['eps'] + names

    for pack in product(*hypers):

        #param['max_depth'] = pack[0]
        #param['eta'] = pack[1]
        param['max_depth'] = depth
        param['eta'] = eta
        param['min_child_weight'] = pack[0]/n
        param['lambda'] = pack[1]
        param['scale_pos_weight'] += pack[2]

        ## PRINT SOME STARTING INFORMATION
        print("START TRIAL")
        print("XGBoost parameters:")
        print(param, flush=True)

        res = dict()

        values = list(pack)
        values.append(seed)
        values.insert(0, eta)
        values.insert(0, depth)
        values.insert(0, eps)

        exp_descriptor = []
        for n, v in zip(names, values):
            exp_descriptor.append(':'.join([n,str(v)]))
        n = Corig.shape[0]
            
        exp_name = '_'.join(exp_descriptor)
        if agebin: exp_name += '_agebin'
        print(exp_name)

        # Use tensorboard to examine information
        fairfile = results_path + exp_name
        if os.path.exists(fairfile):
            shutil.rmtree(fairfile)


        if agebin: fairfile
        summary_writer = tf.summary.create_file_writer(fairfile)
        # Define output metric here so that we can hard code
        # some of the variables (instead of passing a bunch every time we print)
        def tf_metric(bst, dtrain, dtest, max_loss, it, slack=None):
            p0, p1, _ = script.balanced_accuracy(bst, dtest, y_test)
            p0t, p1t, _ = script.balanced_accuracy(bst, dtrain, y_train)

            preds = bst.predict(dtest)
            y_guess = np.array([1 if pval > 0.5 else 0 for pval in preds])

            scons = german_proc.status_cons(bst, X_test, status_inds)
            if agebin:
                abcons = german_proc.flip_cons(bst, X_test, abind)

            if _verbose: print("GENDER GAPS")
            sex_res = script.group_metrics(
                    y_test,
                    y_guess,
                    yt_sex,
                    label_good=1,
                    verbose=_verbose
                    )
            if _verbose: print("AGE GAPS")
            age_res = script.group_metrics(
                    y_test,
                    y_guess,
                    yt_age,
                    label_good=1,
                    verbose=_verbose
                    )

            with summary_writer.as_default():
                tf.summary.scalar('train balanced', (p0t+p1t)/2, step=it)
                tf.summary.scalar('inner training loss', max_loss, step=it)
                tf.summary.scalar('test balanced', (p0+p1)/2, step=it)
                tf.summary.scalar('p0', p0, step=it)
                tf.summary.scalar('p1', p1, step=it)

                tf.summary.scalar(
                        'status cons', scons.sum()/X_test.shape[0], step=it
                        )
                if agebin:
                    tf.summary.scalar(
                            'ab cons', abcons.sum()/X_test.shape[0], step=it
                            )

                tf.summary.scalar('sex-TPR-prot', sex_res[0], step=it)
                tf.summary.scalar('sex-TNR-prot', sex_res[1], step=it)
                tf.summary.scalar('sex-TPR-priv', sex_res[2], step=it)
                tf.summary.scalar('sex-TNR-priv', sex_res[3], step=it)
                tf.summary.scalar('sex-gap-RMS', sex_res[4], step=it)
                tf.summary.scalar('sex-gap-MAX', sex_res[5], step=it)
                tf.summary.scalar('sex-ave-odds-diff', sex_res[6], step=it)
                tf.summary.scalar('sex-eq-opp-diff', sex_res[7], step=it)
                tf.summary.scalar('sex-stat-parity', sex_res[8], step=it)

                tf.summary.scalar('age-TPR-prot', age_res[0], step=it)
                tf.summary.scalar('age-TNR-prot', age_res[1], step=it)
                tf.summary.scalar('age-TPR-priv', age_res[2], step=it)
                tf.summary.scalar('age-TNR-priv', age_res[3], step=it)
                tf.summary.scalar('age-gap-RMS', age_res[4], step=it)
                tf.summary.scalar('age-gap-MAX', age_res[5], step=it)
                tf.summary.scalar('age-ave-odds-diff', age_res[6], step=it)
                tf.summary.scalar('age-eq-opp-diff', age_res[7], step=it)
                tf.summary.scalar('age-stat-parity', age_res[8], step=it)

            summary_writer.flush()

        ## FAIR TRAINING
        t_s = time.time()
        if use_dual:
            print("Using DUAL")
            fst, xport, stuff =  txl.boost_dual(
                    X_train,
                    y_train,
                    C, 
                    n_iter, 
                    X_test=X_test,
                    y_test=y_test,
                    pred=None, 
                    eps=eps, 
                    param=param,
                    verbose=_verbose,
                    verify=True,
                    lowg=0.0,
                    highg=15.0,
                    method='brentq',
                    dtype=_dtype,
                    outfunc=tf_metric,
                    outinterval=1
                    )
        elif not use_dual:
            fst, pi = txl.boost_sinkhorn_tf(
                    X_train,
                    y_train,
                    C,
                    n_iter, 
                    X_test=X_test,
                    y_test=y_test,
                    pred=None, 
                    eps=eps, 
                    gamma_reg=0.00005,
                    param=param,
                    lowg=0.0,
                    highg=5.0,
                    verbose=_verbose,
                    verify=True,
                    roottol=10**(-16),
                    dtype=_dtype,
                    max_float_pow = 65.,
                    outfunc=tf_metric
                    )
        print('Boosting took %f' % (time.time() - t_s))

        final = 0
        if use_dual:
            dists = Corig[xport, np.arange(Corig.shape[1])]

            if stuff is not None:
                dists[stuff[0]] = 0
                final += Corig[stuff[0], stuff[2]]*stuff[3] +\
                        Corig[stuff[0], stuff[1]]*(1/n - stuff[3])

            final += dists.sum()/n
        else:
            final = (Corig*pi.numpy()).sum()

        print("Constraint value: {}".format(final))
        print('DONE')
        
    return 0


if __name__ == '__main__':
    main()
