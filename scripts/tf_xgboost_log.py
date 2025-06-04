# 11 June 2020
# focus on using xgboost with obj=binary:logistic
# We allow for points to move to places with other values.
#
# This file contains the BuDRO boosting implementations 
# using all four methods to solve for the optimal transport solution
# * Dual
# * Dual with SGD
# * Entropic regularization
# * Entropic regularization with SGD
#
# Have fun :)

import tensorflow as tf
import numpy as np
import xgboost as xgb

import tf_roots
import sinkhorn_tf
import sinkhorn_sgd as sgd
import tf_dual
import dual_sgd

import time

def boost_prep(
        X,
        y,
        pred=None,
        bst=None,
        param=None,
        verbose=False
        ):


    # set up xgboost 
    inds0 = np.where(y==0)[0]
    inds1 = np.where(y==1)[0]
    ytrain = np.array([0]*y.shape[0] + [1]*y.shape[0])

    if param is None:
        param = {
                'max_depth':2, 
                'eta':0.1, 
                'objective':'binary:logistic', 
                'min_child_weight':2/X.shape[0],
                'lambda':0.2
                }

        if verbose:
            print("Warning: Input xgboost parameters not found")
            print("Warning: Using default parameters", flush=True)

    
    # Start with one step of naive boosting if no initial guess
    dorig = xgb.DMatrix(data=X, label=y)
    res = dict()
    if pred is None and bst is None:

        if verbose:
            print("Warning: No input probabilities " +\
                "for an original classifier")
            print("Warning: Starting with one step of naive boosting")


        bst = xgb.train(
                param,
                dorig,
                num_boost_round=1,
                evals=[(dorig, 'orig')],
                evals_result=res,
                verbose_eval=verbose
                )

    return inds0, inds1, ytrain, param, bst, dorig, res

#def write_data_to_tb(
#        writer,
#        max_loss,
#        bst,
#        dtrain,
#        dtest,
#        X_test,
#        y_sex_test,
#        y_race_test,
#        feature_names
#        ):


# A simple standard sinkhorn tf booster
# Can't use this if you need a lot of memory
#
# preds should be original probability of having label 1 (according to the
# first classifier).  if you don't have it, we will eventually just fit one
# without it.  This is specifically for binary classification with the log
# loss.
#
# Expect that C is a tf.constant
# Only using tensorflow for the attack step - most calculations done
# in numpy
def boost_sinkhorn_tf(
        X, y, C, n_iter, 
        X_test=None,
        y_test=None,
        pred=None, 
        eps=0.01, 
        gamma_reg=0.1,
        param=None,
        bst=None,
        lowg=0.0,
        highg=5.0,
        verbose=False,
        verify=False,
        roottol=10**(-16),
        dtype=tf.float32,
        max_float_pow = 65.,
        outinterval=10,
        outfunc=None
        ):

    n=int(C.shape[0])

    # set up xgboost parameters
    inds0, inds1, ytrain, param, bst, dorig, res =\
            boost_prep(X, y, pred, bst, param, verbose)

    test_data = X_test is not None and y_test is not None
    if test_data:
        dtest = xgb.DMatrix( data=X_test, label=y_test )

    num_round=1
    res = dict()

    # set up tensorflow
    #print("Making tensorflow eps and gamma_reg", flush = True)
    eps = tf.constant(eps, dtype=dtype)
    gamma_reg = tf.constant(gamma_reg, dtype=dtype)

    for it in range(n_iter):
        print("Iter {}:".format(it), end=" ")

        # update losses
        if verbose: print("Calculate losses and make L matrix")

        if bst is not None:
            pred = bst.predict(dorig).astype('float64', casting='safe')

        losses0 = - np.log(1-pred)
        losses1 = - np.log(pred)

        if verbose: print("Calculating transport map with sinkhorn", flush=True)
        # find transport matrix
        pij, _ = tf_roots.sinkhorn_tf(
                C,
                tf.constant(losses0, dtype),
                tf.constant(losses1, dtype),
                y, # y is always cast in sinkhorn_tf
                eps,
                gamma_reg,
                lowg=lowg,
                highg=highg,
                dtype=dtype,
                max_float_pow=max_float_pow,
                verbose=verbose,
                verify=verify
                )

        wts0tf = tf.reduce_sum(
                tf.gather(tf.transpose(pij), inds0),
                axis=0
                )
        wts1tf = tf.reduce_sum(
                tf.gather(tf.transpose(pij), inds1),
                axis=0
                )


        # kill the smallest values
        # In my experience, the largest difference between
        # weights, repeating the same computation twice, is about 10**-9
        wts0 = wts0tf.numpy()
        wts0[wts0 < roottol] = 0
        wts1 = wts1tf.numpy()
        wts1[wts1 < roottol] = 0

        if verbose: print("Boost for a step", flush=True)
        # boost for a step
        dtrain = xgb.DMatrix(
            data=np.concatenate((X,X)), 
            label=ytrain, 
            weight=np.concatenate((wts0, wts1))
        )

        watchlist = [(dtrain, 'double'), (dorig, 'orig')]
        if test_data is True:
            watchlist += [(dtest, 'test')]


        bst = xgb.train(
                param, 
                dtrain, 
                num_round, 
                watchlist, 
                xgb_model=bst, 
                evals_result=res,
                verbose_eval=verbose
                )

        if it % outinterval == outinterval - 1:
            if outfunc is not None:
                max_loss = wts0.dot(losses0) + wts1.dot(losses1)
                outfunc(bst, dtrain, dtest, max_loss, it)
        

    return bst, pij

# Simple sinkhorn boosting to be used on 2 gpus.  Can handle higher memory.
# We use the sinkhorn class here. 
#
# preds should be original probability of having label 1 (according to the
# first classifier).  if you don't have it, we will eventually just fit one
# without it.  This is specifically for binary classification with the log
# loss.
#
# Input an already created sinkhorn object (with the graph traced out)
def boost_sinkhorn_2gpu(
        X, y, sinkhorn, n_iter, 
        X_test=None,
        y_test=None,
        pred=None, 
        eps=0.01, 
        gamma_reg=0.1,
        param=None,
        bst=None,
        lowg=0.0,
        highg=5.0,
        verbose=False,
        roottol=10**(-16),
        dtype=tf.float32,
        idtype=tf.int32
        ):

    #print("In boost_sinkhorn_2gpu", flush=True)
    n=int(C.shape[0])

    # set up xgboost parameters
    inds0, inds1, ytrain, param, bst, dorig, res =\
            boost_prep(X, y, pred, bst, param, verbose)

    test_data = X_test is not None and y_test is not None
    if test_data:
        dtest = xgb.DMatrix( data=X_test, label=y_test )

    num_round=1
    res = dict()

    # set up tensorflow
    #print("Making tensorflow eps and gamma_reg", flush = True)
    eps = tf.constant(eps, dtype=sinkhorn.dtype)
    gamma_reg = tf.constant(gamma_reg, dtype=sinkhorn.dtype)

    for it in range(n_iter):
        print("Iter {}:".format(it), end=" ")

        # update losses
        if verbose: print("Calculate losses")

        if bst is not None:
            pred = bst.predict(dorig).astype('float64', casting='safe')

        losses0 = - np.log(1-pred)
        losses1 = - np.log(pred)

        if verbose: print("Calculating transport map with sinkhorn", flush=True)
        # find transport matrix
        pij = sinkhorn_pi_2gpu(
                sinkhorn,
                tf.constant(losses0, dtype=sinkhorn.dtype),
                tf.constant(losses1, dtype=sinkhorn.dtype),
                eps,
                gamma_reg,
                lowg = lowg,
                highg = highg,
                verbose=verbose
                )


        # you might need to fux with this to use multiple GPUs
        with tf.device('/GPU:1'):
            wts0tf = tf.reduce_sum(
                    #tf.gather(tf.transpose(pij), tf.cast(inds0, idtype)),
                    tf.gather(tf.transpose(pij), inds0),
                    axis=0
                    )
            wts1tf = tf.reduce_sum(
                    tf.gather(tf.transpose(pij), inds1),
                    axis=0
                    )

        # kill the smallest values
        # In my experience, the largest difference between
        # weights, repeating the same computation twice, is about 10**-9
        wts0 = wts0tf.numpy()
        wts0[wts0 < roottol] = 0
        wts1 = wts1tf.numpy()
        wts1[wts1 < roottol] = 0

        if verbose: print("Boost for a step", flush=True)
        # boost for a step
        dtrain = xgb.DMatrix(
            data=np.concatenate((X,X)), 
            label=ytrain, 
            weight=np.concatenate((wts0, wts1))
        )

        watchlist = [(dtrain, 'double'), (dorig, 'orig')]
        if test_data is True:
            watchlist += [(dtest, 'test')]

        bst = xgb.train(
                param, 
                dtrain, 
                num_round, 
                watchlist, 
                xgb_model=bst, 
                evals_result=res,
                verbose_eval=verbose
                )
        
        # clear memory when not on the last step
        # My experience is that garbage collection is automatic (don't need
        # the last line but I'm keeping it there just in case)
        if (it != n_iter - 1):
            del pij
            del wts0tf
            del wts1tf
            #tf.keras.backend.clear_session()

    return bst, pij

def sinkhorn_pi_2gpu(
        sinkhorn,
        tfl0,
        tfl1,
        eps,
        gamma_reg,
        lowg=0.0,
        highg=4.0,
        verbose=True
        ):

    if(verbose):
        print("***Attack step: finding root eta.", flush=True)
        t_s = time.time()

    sol = sinkhorn.find_eta(tfl0, tfl1, eps, gamma_reg, lowg=lowg, highg=highg)

    if (verbose):
        print('Took %f' % (time.time() - t_s))
        print("optimal value of eta: {}".format(sol.root))
        print("Further solution information")
        print(sol)

    if(verbose):
        print("***Attack step: constructing pi on GPU:1.", flush=True)
        t_s = time.time()

    with tf.device('/GPU:1'):
        pi = sinkhorn.make_pi(
                tfl0, 
                tfl1, 
                tf.constant(sol.root, dtype=sinkhorn.dtype), 
                gamma_reg
                )

    return pi

# Double gpu sgd booting.
#
# preds should be original probability of having label 1 (according to the
# first classifier).  if you don't have it, we will eventually just fit one
# without it.  This is specifically for binary classification with the log
# loss.
def boost_sinkhorn_sgd_2gpu(
        X, y, Corig, C, n_iter, 
        tf_eta,
        X_test=None,
        y_test=None,
        pred=None, 
        eps=0.01, 
        gamma_reg=0.1,
        epoch=100,
        batch_size=100,
        lr=0.001,
        momentum=0.9,
        init=0.1,
        param=None,
        bst=None,
        verbose=False,
        roottol=10**(-16),
        dtype=tf.float32,
        max_float_pow = 65.,
        outinterval=10,
        outfunc=None
        ):

    n=Corig.shape[0]

    # set up xgboost parameters
    inds0, inds1, ytrain, param, bst, dorig, res =\
            boost_prep(X, y, pred, bst, param, verbose)

    test_data = X_test is not None and y_test is not None
    if test_data:
        dtest = xgb.DMatrix( data=X_test, label=y_test )

    num_round=1
    res = dict()

    # set up tensorflow
    #print("Making tensorflow eps and gamma_reg", flush = True)
    eps = tf.constant(eps, dtype=dtype)
    gamma_reg = tf.constant(gamma_reg, dtype=dtype)

    for it in range(n_iter):
        print("Iter {}:".format(it), end=" ")

        # update losses
        if verbose: print("Calculate losses")

        if bst is not None:
            pred = bst.predict(dorig).astype('float64', casting='safe')

        losses0 = - np.log(1-pred)
        losses1 = - np.log(pred)

        if verbose: print("Calculating transport map with sinkhorn", flush=True)
        # find transport matrix
        pij = sgd.sinkhorn_sgd_pi_2gpu(
                tf_eta,
                Corig,
                y,
                C,
                tf.constant(losses0, dtype=dtype),
                tf.constant(losses1, dtype=dtype),
                eps,
                gamma_reg,
                epoch,
                batch_size,
                n=n,
                dtype=dtype,
                verbose=verbose,
                lr=lr,
                momentum=momentum,
                init=init,
                newSoln=True,
                max_float_pow = max_float_pow
                )


        with tf.device('/GPU:1'):
            wts0tf = tf.reduce_sum(
                    tf.gather(tf.transpose(pij), inds0),
                    axis=0
                    )
            wts1tf = tf.reduce_sum(
                    tf.gather(tf.transpose(pij), inds1),
                    axis=0
                    )

        # kill the smallest values
        # In my experience, the largest difference between
        # weights, repeating the same computation twice, is about 10**-9
        #
        # This probably doesn't take too long, so I am leaving it
        # (even though I am normalizing, which seems to round small values
        # towards 0).
        wts0 = wts0tf.numpy()
        wts0[wts0 < roottol] = 0
        wts1 = wts1tf.numpy()
        wts1[wts1 < roottol] = 0

        if verbose: print("Boost for a step", flush=True)
        # boost for a step
        dtrain = xgb.DMatrix(
            data=np.concatenate((X,X)), 
            label=ytrain, 
            weight=np.concatenate((wts0, wts1))
        )

        watchlist = [(dtrain, 'double'), (dorig, 'orig')]
        if test_data is True:
            watchlist += [(dtest, 'test')]

        bst = xgb.train(
                param, 
                dtrain, 
                num_round, 
                watchlist, 
                xgb_model=bst, 
                evals_result=res,
                verbose_eval=verbose
                )

        if it % outinterval == outinterval - 1:
            if outfunc is not None:
                max_loss = wts0.dot(losses0) + wts1.dot(losses1)
                outfunc(bst, dtrain, dtest, max_loss, it)
        
        # clear memory when not on the last step
        # My experience is that garbage collection is automatic (don't need
        # the last line but I'm keeping it there just in case)
        if (it != n_iter - 1):
            del pij
            del wts0tf
            del wts1tf
            #tf.keras.backend.clear_session()

    return bst, pij

def boost_dual_sgd(
        X, y, Corig, C, n_iter, 
        tf_eta,
        X_test=None,
        y_test=None,
        pred=None, 
        eps=0.01, 
        epoch=100,
        batch_size=100,
        lr=0.001,
        momentum=0.9,
        init=0.1,
        param=None,
        bst=None,
        verbose=False,
        verify=False,
        dtype=tf.float32,
        lowpimem=False,
        outinterval=10,
        outfunc=None
        ):

    n=Corig.shape[0]

    # set up xgboost parameters
    inds0, inds1, ytrain, param, bst, dorig, res =\
            boost_prep(X, y, pred, bst, param, verbose)

    test_data = X_test is not None and y_test is not None
    if test_data:
        dtest = xgb.DMatrix( data=X_test, label=y_test )

    num_round=1
    res = dict()

    # set up tensorflow
    eps = tf.constant(eps, dtype=dtype)

    for it in range(n_iter):
        print("Iter {}:".format(it), end=" ")

        # update losses
        if verbose: print("Calculate losses")

        if bst is not None:
            pred = bst.predict(dorig).astype('float64', casting='safe')

        losses0 = - np.log(1-pred)
        losses1 = - np.log(pred)

        if verbose: print("Calculating transport map with sinkhorn", flush=True)
        # find transport matrix
        wts0, wts1, xport_inds = dual_sgd.dual(
                tf_eta,
                Corig, 
                y,
                C,
                tf.constant(losses0, dtype=dtype),
                tf.constant(losses1, dtype=dtype),
                eps,
                epoch, 
                batch_size, 
                dtype=dtype,
                idtype=tf.int32,
                verbose = verbose,
                verify=verify,
                lr=lr,
                momentum=momentum,
                init=0.1,
                newSoln=True,
                lowpimem=lowpimem,
                )

        if verbose: print("Boost for a step", flush=True)
        # boost for a step
        dtrain = xgb.DMatrix(
            data=np.concatenate((X,X)), 
            label=ytrain, 
            weight=np.concatenate((wts0, wts1))
        )

        watchlist = [(dtrain, 'double'), (dorig, 'orig')]
        if test_data is True:
            watchlist += [(dtest, 'test')]

        bst = xgb.train(
                param, 
                dtrain, 
                num_round, 
                watchlist, 
                xgb_model=bst, 
                evals_result=res,
                verbose_eval=verbose
                )

        # print on the first step, every outinterval steps, and the last step
        if outfunc is not None:
            if it % outinterval == 0 or it == n_iter - 1:
                max_loss = wts0.dot(losses0) + wts1.dot(losses1)
                opt_dists = Corig[xport_inds, np.arange(Corig.shape[1])]
                slack = eps - opt_dists.mean()
                outfunc(bst, dtrain, dtest, max_loss, it, slack=slack)

    return bst, xport_inds

def boost_dual(
        X, y, C, n_iter, 
        X_test=None,
        y_test=None,
        pred=None, 
        eps=0.01, 
        param=None,
        bst=None,
        verbose=False,
        verify=False,
        lowg=0.0,
        highg=10.0,
        method='brentq',
        roottol=10**(-8),
        dtol=10**(-25),
        dtype=tf.float32,
        idtype=tf.int32,
        outinterval=10,
        outfunc=None
        ):

    n=int(C.shape[0])

    # set up xgboost parameters
    inds0, inds1, ytrain, param, bst, dorig, res =\
            boost_prep(X, y, pred, bst, param, verbose)

    test_data = X_test is not None and y_test is not None
    if test_data:
        dtest = xgb.DMatrix( data=X_test, label=y_test )

    num_round=1
    res = dict()

    # set up tensorflow
    eps = tf.constant(eps, dtype=dtype)

    for it in range(n_iter):
        print("Iter {}:".format(it), end=" ")

        # update losses
        if verbose: print("Calculate losses")

        if bst is not None:
            pred = bst.predict(dorig).astype('float64', casting='safe')

        losses0 = - np.log(1-pred)
        losses1 = - np.log(pred)

        if verbose: print("Calculating transport map with dual", flush=True)
        # find transport matrix
        wts0, wts1, sol, xport_inds, stuff = tf_dual.find_wts(
                tf.constant(losses0, dtype=dtype),
                tf.constant(losses1, dtype=dtype),
                y,
                C,
                eps, 
                lowg=lowg, 
                highg=highg, 
                method=method, 
                roottol=roottol,
                dtol=dtol,
                dtype=dtype,
                idtype=idtype,
                debug=verify,
                verbose=verbose
                )

        if verbose: print("Boost for a step", flush=True)
        # boost for a step
        dtrain = xgb.DMatrix(
            data=np.concatenate((X,X)), 
            label=ytrain, 
            weight=np.concatenate((wts0, wts1))
        )

        watchlist = [(dtrain, 'double'), (dorig, 'orig')]
        if test_data is True:
            watchlist += [(dtest, 'test')]

        bst = xgb.train(
                param, 
                dtrain, 
                num_round, 
                watchlist, 
                xgb_model=bst, 
                evals_result=res,
                verbose_eval=verbose
                )

        if it % outinterval == outinterval - 1:
            if outfunc is not None:
                max_loss = wts0.dot(losses0) + wts1.dot(losses1)
                outfunc(bst, dtrain, dtest, max_loss, it)

    return bst, xport_inds, stuff
