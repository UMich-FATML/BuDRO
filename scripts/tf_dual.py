import tensorflow as tf
import numpy as np
from scipy.optimize import root_scalar

# finding the dual optimal without SGD using tensorflow


@tf.function
def dual_tf(eta, L, C, eps, dtype=tf.float64):
    return eta*eps + tf.reduce_sum(
            tf.reduce_max(
                tf.add(L, tf.scalar_mul(-eta, C)),
            axis=0 )
        )/C.shape[0]


# Timing: n = 10000, ~0.166 seconds per iteration for 100 iterations
@tf.function
def dual_inds_tf(eta, L, C, dtype=tf.float64, idtype=tf.int64):
    return tf.argmax(
                L - tf.scalar_mul(tf.cast(eta, dtype=dtype),C), 
                axis=0,
                output_type=idtype
            )

# Both of the following two functions seem to take the same amount of time
# on a subset of adult with n = 10000 (~0.172 seconds for f_grad, ~0.166 seconds
# for f_grad_tf, average based on 100 iterations)
# Thus, the dual_inds_tf function is taking the majority of the time either way.
#
# Root scalar with maxiter=30 and bracket=[0,5] takes ~5.3 seconds/iteration

# Not a tf.function: check reducing outside of tensorflow
def f_grad(eta, L, C, Corig, eps, dtype=tf.float64):
    inds = dual_inds_tf(eta, L, C, dtype=dtype)
    return tf.cast(eps, dtype=dtype) - Corig[ 
                inds.numpy(), 
                np.arange(Corig.shape[1]) 
            ].sum()/Corig.shape[0]

@tf.function
def f_grad_tf(eta, L, C, eps, dtype=tf.float64, idtype=tf.int64):

    inds = dual_inds_tf(eta, L, C, dtype, idtype)

    inner = tf.reduce_sum(
            tf.gather_nd(
                C,
                tf.transpose(
                    tf.stack((
                        tf.range(C.shape[0], dtype=idtype),
                        inds
                    ))
                )
            ))

    return tf.cast(eps, dtype=dtype) - inner/C.shape[0]


# About 4.1 seconds per iteration when n=10000
# Use sgd if you want to get larger
def find_wts(
        loss0, loss1, y, C, eps, 
        lowg=0.0, 
        highg=10.0, 
        method='brentq', 
        roottol=10**(-8),
        dtol=10**(-25),
        dtype=tf.float32,
        idtype=tf.int32,
        debug=False,
        verbose=False
        ):


    # basic parameter set up
    n = y.shape[0]
    ytf = tf.constant(y, dtype=dtype)
    inds0 = np.where(y==0)[0]
    inds1 = np.where(y==1)[0]

    # make L with kronecker products in tensorflow
    L = tf.transpose(
            tf.tensordot(tf.cast(ytf, dtype=dtype), loss1, axes=0) +\
            tf.tensordot(tf.cast(1-ytf, dtype=dtype), loss0, axes=0)
        )

    # compute initial value of inner stuff
    if debug:
        obj0 = loss0.numpy().dot(1-y) + loss1.numpy().dot(y)
        obj0 = obj0/n

    # This takes ~10-11 seconds for 3 iterationsin my example with the
    # newmetric (sol is ~0.4)
    sol = find_eta(C, L, eps, lowg, highg, method, dtype, idtype)

    # could make a temporary matrix here if want to speed up more
    # values of dual variables
    # testing has this step at .08 seconds when n = 10000
    lambs = tf.reduce_max(
                L -  tf.scalar_mul(tf.cast(sol.root, dtype=dtype), C),
                axis=0
            )

    # zeros here are the entries of pi that could be nonzero
    zvals = L - tf.scalar_mul(tf.cast(sol.root, dtype=dtype), C) -lambs
    
    # Check your tolerances - are you doing things correctly
    if debug and verbose:
        error = tf.reduce_max(tf.abs(
                    tf.gather_nd(zvals, tf.where(tf.abs(zvals) <= roottol))
                ))
        if error > 0:
            print("Debug: Largest difference of dual constraint from the optimal")
            print("Debug: smaller than tolerance {}:".format(roottol))
            print("Debug: {}".format(error), flush=True)

    # next five steps are very fast
    # Find the spots that are smaller than the input tolerance
    spots = tf.cast(
            tf.where( tf.abs(zvals) <= tf.cast(roottol, dtype=dtype) ),
            dtype=idtype
            )

    # reorder by column 
    spots = tf.gather(spots, tf.argsort(spots[:,1]))

    # Get splitting information
    _, _, split = tf.unique_with_counts(tf.sort(spots[:,1]))

    # delete column information because we will track of it ourselves
    xports = spots[:,0]

    xportData = tf.RaggedTensor.from_row_lengths(
                values = xports,
                row_lengths = split
            )

    cData = tf.RaggedTensor.from_row_lengths(
                values = tf.gather_nd(C, spots),
                row_lengths = split
            )

    # takes ~0.1 seconds for n = 10000
    # dubs is the column that is split.
    ad = -tf.reduce_min(cData, axis=1)
    ad = tf.transpose(ad[None,:])

    dubs = tf.where(tf.reduce_any(cData + ad > dtol, axis=1)).numpy().T[0]

    split = False
    col = None

    if dubs.shape[0] > 0:
        split = True
        col = dubs[0]

    if dubs.shape[0] > 1:
        print(
            "Warning: Multiple points have more "+\
            "than one non-trivial transport location."
        )
        print("Warning: All but first point ignored", flush=True)
        print("Warning: Points are {}".format(dubs))
        #print("Cost values:")
        #for ind in dubs:
        #    print("{}:  {}".format(ind,cData[ind]))

        if debug:
            dupCosts = tf.gather(tf.transpose(tf.gather(C, dubs)), dubs)
            print("Warning: Maximum distance between these points: {}".format(
                tf.reduce_max(dupCosts)))

    # is the split in inds0 or inds1?
    colWt = None
    if col in inds0: colWt = 0
    elif col in inds1: colWt = 1

    # delete col from the inds
    inds0 = inds0[inds0!=col]
    inds1 = inds1[inds1!=col]

    # make the weights that are directly transported
    wts0 = np.zeros(n)
    wts1 = np.zeros(n)
    
    xport_map = xportData[:,0:1].values.numpy()
    dists = cData[:,0:1].values.numpy()

    res = dists[inds0].sum() + dists[inds1].sum()

    for ind in xport_map[inds0]: wts0[ind] += 1
    for ind in xport_map[inds1]: wts1[ind] += 1

    wts0 = wts0/n
    wts1 = wts1/n
    res = res/n
    resid = eps.numpy() - res
    
    # Not checking for more than two transport spots now
    split_data = None
    if split:
        oneData = cData[col]

        # minInd will correspond to col if we want to stay home
        minInd = tf.argmin(oneData)
        maxInd = tf.where(oneData > oneData[minInd] + dtol)[0,0]

        port0 = xportData[col, minInd]
        port1 = xportData[col, maxInd]

        c0 = C[col, port0]
        c1 = C[col, port1]

        piVal = np.min([1/n, (resid - c0/n)/(c1 - c0)])

        if colWt == 0:
            wts0[port1] += piVal
            wts0[port0] += 1/n - piVal
        if colWt == 1:
            wts1[port1] += piVal
            wts1[port0] += 1/n - piVal

        split_data = [col, port0, port1, piVal] 

    if debug:
        objf = loss0.numpy().dot(wts0) + loss1.numpy().dot(wts1)
        print("Final value of inner max: {}".format(objf))

        diff = objf-obj0
        if diff < 0:
            print("Warning: sinkhorn attack failed to increase loss objective.", 
                    flush=True)

    return wts0, wts1, sol, xport_map, split_data

def find_eta(
        C,L,eps,
        lowg=0.0,
        highg=10.0,
        method='brentq',
        dtype=tf.float32,
        idtype=tf.int32
        ):

    # find dual solution
    # Not sure if I can tf.function this
    obj_f = lambda x: f_grad_tf(
            tf.constant(x, dtype=dtype),
            L,
            C,
            eps,
            dtype=dtype,
            idtype=idtype
            )

    sol = root_scalar(obj_f, bracket=[lowg, highg], method=method)
    return sol
