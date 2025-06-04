import tensorflow as tf
import numpy as np
import time

# Functions for finding dual optimizer using sgd
# Implemented with tensorflow

# We create a batch L online to avoid needed to construct L with numpy 
# (which takes about 30 seconds and needs to be done every iteration).
# 
# We would need to construct L with numpy to save memory, though I suppose
# we can easily use float32 in this case
#
# But the creation time is pretty quick (much less than a second) if I remember
# correctly.  Looking it up: around 0.04 seconds to construct the full L.
# That means we need 750 generations of the full L to make up for things.
# We are expecting around 30 calls per boosting step.

# not messing around with casting or protecting dtypes or anything.
# Be careful with your inputs
@tf.function
def dual_obj(
        tf_eta,
        C_batch,
        loss0,
        loss1,
        y_batch,
        eps,
        ):

    return tf_eta*eps + tf.reduce_mean(
            tf.reduce_max(
                tf.tensordot(y_batch, loss1, axes=0) +\
                tf.tensordot(1 - y_batch, loss0, axes=0) -\
                tf.scalar_mul(tf_eta, C_batch),
                axis=1
                )
            )


# make L on the fly to save memory - can maybe do this on
# GPU 1 if needed.
@tf.function
def dual_inds(
        eta,
        C,
        loss0,
        loss1,
        ytf,
        idtype=tf.int64
        ):

    return tf.argmax(
            tf.tensordot(ytf, loss1, axes=0) +\
            tf.tensordot(1 - ytf, loss0, axes=0) -\
            tf.scalar_mul(eta,C), 
            axis=1,
            output_type=idtype
            )

def dual_sgd(
        tf_eta,
        Corig, 
        loss0,
        loss1,
        y,
        eps,
        epoch, 
        batch_size, 
        dtype=tf.float32,
        verbose=False,
        lr=0.05,
        momentum=0.9,
        ):

    n = Corig.shape[0]

    batch_indices = np.random.choice(n, size=(epoch,batch_size))

    optimizer = tf.keras.optimizers.SGD(
                learning_rate=lr, 
                momentum=momentum
            )

    for i in tf.range(epoch):

        inds = batch_indices[i]

        optimizer.minimize(
            loss = lambda: dual_obj(
                tf_eta,
                tf.constant(Corig[inds], dtype=dtype),
                loss0,
                loss1,
                tf.constant(y[inds], dtype=dtype),
                eps,
            ),
            var_list = [tf_eta]
        )

        if verbose and i % 10 == 0:
            print('Current eta: {}'.format(tf_eta.numpy()))

    eta = tf_eta.numpy()

    return eta 

# Really no seatbelts for this right now.
# lowmem will fail if you use too large of a data set.
def dual_weights(
        eta,
        C,
        loss0,
        loss1,
        y,
        n=None,
        lowpimem=False,
        num_gbs=2,
        dtype=tf.float32,
        idtype=tf.int32,
        ):

    if n is None:
        n = C.shape[0]

    # could maybe do this in a for loop in a tf.function
    # like do a dual_inds_lowmem
    # but whatever for now
    if lowpimem:

        rows_in_gb = np.floor(num_gbs * 2**30/(4*n))
        inds = int(rows_in_gb)*np.array(range( int(np.ceil( n/rows_in_gb )) + 1 ))
        inds[-1] = n

        max_inds_list = []

        for i in range(inds.shape[0] - 1):
            max_inds_list.append(
                    dual_inds(
                        eta,
                        C[inds[i]:inds[i+1]],
                        loss0,
                        loss1,
                        tf.constant(y[inds[i]:inds[i+1]], dtype),
                        idtype
                        )
                    )

        max_inds = np.concatenate(max_inds_list)

    else: # Use whatever memory we want

        max_inds = dual_inds(
                    eta,
                    C,
                    loss0,
                    loss1,
                    tf.constant(y, dtype),
                    idtype
                    ).numpy()

    inds0 = max_inds[y==0]
    inds1 = max_inds[y==1]

    wts0 = np.zeros(y.shape[0])
    wts1 = np.zeros(y.shape[0])

    for i in inds0:
        wts0[i] += 1

    for i in inds1:
        wts1[i] += 1

    wts0 = wts0/n
    wts1 = wts1/n

    return wts0, wts1, max_inds

def dual(
        tf_eta,
        Corig, 
        y,
        C,
        loss0,
        loss1,
        eps,
        epoch, 
        batch_size, 
        dtype=tf.float32,
        idtype=tf.int32,
        verbose =False,
        verify=False,
        lr=0.05,
        momentum=0.9,
        init=1.0,
        newSoln=True,
        lowpimem=False,
        ):

    if newSoln: tf_eta.assign(init)

    n = Corig.shape[0]

    if verify:
        obj0 = (loss0.numpy()[y==0].sum() + loss1.numpy()[y==1].sum())/n
        if verbose:
            print("Original loss: {}".format(obj0), flush=True)

    if(verbose):
        print("***Dual attack step: finding optimum eta.", flush=True)
        t_s = time.time()

    eta = dual_sgd(
        tf_eta,
        Corig, 
        loss0,
        loss1,
        y,
        eps,
        epoch, 
        batch_size, 
        dtype=dtype,
        verbose=verbose,
        lr=lr,
        momentum=momentum,
        )

    if (verbose):
        print('Took %f' % (time.time() - t_s))
        print("discovered value of eta: {}".format(eta))

        print("***Dual attack step: finding weights.", flush=True)
        t_s = time.time()

    wts0, wts1, max_inds = dual_weights(
            tf.constant(eta, dtype=dtype),
            C,
            loss0,
            loss1,
            y,
            n=n,
            lowpimem=lowpimem,
            dtype=dtype,
            idtype=idtype,
            )

    if verify:
        opt_dists = Corig[max_inds, np.arange(Corig.shape[1])]
        slack = eps - opt_dists.mean()
        if verbose: print("Dual slackness: {}".format(slack), flush=True)

        objf = wts0.dot(loss0) + wts1.dot(loss1)
        print("Final value of inner max: {}".format(objf), flush=True)

        diff = objf-obj0
        if verbose:
            print("Change in max: {}".format( diff ))

        if diff < 0:
            print("Warning: sinkhorn attack failed to increase loss objective.", 
                    flush=True)

    return wts0, wts1, max_inds
