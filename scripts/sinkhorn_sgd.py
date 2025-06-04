
import numpy as np
import tensorflow as tf
import time

_max_float_pow = 65.

# Finding the optimal transport solution using the sinkhorn method with sgd
# Implemented with tensorflow 2.0
# default _max_float_pow assumes working with float32.

@tf.function
def loss_lowmem(
        tf_eta,
        batch_C,
        loss0,
        loss1,
        y_batch,
        eps,
        gamma_reg,
        dtype=tf.float64,
        ):

    # Calculate L on the fly based on the input y
    exp_term = tf.reduce_sum(tf.exp(
                tf.scalar_mul(
                    tf.cast(1.0/gamma_reg, dtype=dtype),
                    tf.tensordot(tf.cast(y_batch, dtype=dtype), loss1, axes=0) +\
                    tf.tensordot(tf.cast(1-y_batch, dtype=dtype), loss0, axes=0)-\
                    tf_eta * batch_C
                )), 
                axis=1
            )

    return eps*tf_eta + gamma_reg * tf.reduce_mean(tf.math.log(exp_term))

@tf.function
def loss_norm_lowmem(
        tf_eta,
        batch_C,
        loss0,
        loss1,
        y_batch,
        eps,
        gamma_reg,
        dtype=tf.float64,
        max_float_pow = _max_float_pow
        ):

    exp_term = tf.scalar_mul(
            tf.cast(1.0/gamma_reg, dtype=dtype),
            tf.tensordot(tf.cast(y_batch, dtype=dtype), loss1, axes=0) +\
            tf.tensordot(tf.cast(1-y_batch, dtype=dtype), loss0, axes=0)-\
            tf_eta * batch_C
        )

    t_max = tf.reduce_max(exp_term, axis=1)
    t_max = tf.multiply(
            t_max - max_float_pow,
            tf.cast(t_max > max_float_pow, dtype=dtype)
        )

    exp_term = tf.reduce_sum(
            tf.exp( exp_term - t_max[:,None] ),
            axis=1
        )

    return eps*tf_eta + gamma_reg * tf.reduce_mean(tf.math.log(exp_term) + t_max)



@tf.function
def loss(
        tf_eta,
        C_batch,
        L_batch,
        eps,
        gamma_reg,
        dtype=tf.float64,
        ):

    exp_term = tf.reduce_sum(tf.exp(
                tf.scalar_mul(
                    tf.cast(1/gamma_reg, dtype=dtype),
                    L_batch - tf_eta * C_batch
                )), 
                axis=1,
            )

    return eps*tf_eta + gamma_reg * tf.reduce_mean(tf.math.log(exp_term))


# As written, we still need to store L, C, and temp on the GPU
# So this isn't really a low-memory implementation yet.
# Could compute temp in numpy for lower memory
def sinkhorn_sgd(
        tf_eta,
        C, 
        loss0,
        loss1,
        y,
        eps,
        gamma_reg,
        epoch, 
        batch_size, 
        n = None,
        dtype=tf.float64,
        idtype=tf.int64,
        verbose = True,
        lr=0.05,
        momentum=0.9
        ):


    if n is None:
        n = C.shape[0]

    # quick construction of L using tensorflow
    L = tf.tensordot(tf.cast(y, dtype=dtype), loss1, axes=0) +\
        tf.tensordot(tf.cast(1-y, dtype=dtype), loss0, axes=0)

    batch_indices = np.random.choice(n, size=(epoch,batch_size))

    optimizer = tf.keras.optimizers.SGD(
                learning_rate=lr, 
                momentum=momentum
            )

    for i in tf.range(epoch):

        inds = batch_indices[i]

        optimizer.minimize(
            loss = lambda: loss(
                tf_eta,
                tf.gather(C,inds), 
                tf.gather(L,inds),
                eps,
                gamma_reg,
                dtype
            ),
            var_list = [tf_eta]
        )

        if verbose and i % 10 == 0:
            print('Current eta: {}'.format(tf_eta.numpy()))

    eta = tf_eta.numpy()
    temp = tf.exp( tf.transpose(L)/gamma_reg -\
            tf.scalar_mul(tf.cast(eta/gamma_reg, dtype=dtype), C)
            )
    u = tf.math.reciprocal( 
        tf.scalar_mul(
            tf.cast(C.shape[0], dtype), 
            tf.reduce_sum(temp, 0) 
        ) 
    )

    return tf.multiply(temp, u), eta 

# Using lower memory on the GPU 
# Don't store the matrix L
# Don't find the final solution Pi (leave that to a different subroutine)
#
# With batch_size = 100, this takes ~0.02 seconds per iteration.  Still might
# be possible to speed this up.   That would help.
#
# Corig should be a numpy array
def sinkhorn_sgd_lowmem(
        tf_eta,
        Corig, 
        loss0,
        loss1,
        y,
        eps,
        gamma_reg,
        epoch, 
        batch_size, 
        n = None,
        dtype=tf.float64,
        idtype=tf.int64,
        verbose = True,
        lr=0.05,
        momentum=0.9
        ):


    if n is None:
        n = C.shape[0]

    batch_indices = np.random.choice(n, size=(epoch,batch_size))

    optimizer = tf.keras.optimizers.SGD(
                learning_rate=lr, 
                momentum=momentum
            )

    for i in tf.range(epoch):

        inds = batch_indices[i]

        optimizer.minimize(
            loss = lambda: loss_lowmem(
                tf_eta,
                tf.constant(Corig[inds], dtype=dtype),
                loss0,
                loss1,
                tf.constant(y[inds], dtype=dtype),
                eps,
                gamma_reg,
                dtype
            ),
            var_list = [tf_eta]
        )

        if verbose and i % 10 == 0:
            print('Current eta: {}'.format(tf_eta.numpy()))

    eta = tf_eta.numpy()

    return eta 

def sinkhorn_sgd_norm_lowmem(
        tf_eta,
        Corig, 
        loss0,
        loss1,
        y,
        eps,
        gamma_reg,
        epoch, 
        batch_size, 
        n = None,
        dtype=tf.float64,
        verbose = True,
        lr=0.05,
        momentum=0.9,
        max_float_pow = _max_float_pow
        ):


    if n is None:
        n = C.shape[0]

    batch_indices = np.random.choice(n, size=(epoch,batch_size))

    optimizer = tf.keras.optimizers.SGD(
                learning_rate=lr, 
                momentum=momentum
            )

    for i in tf.range(epoch):

        inds = batch_indices[i]

        optimizer.minimize(
            loss = lambda: loss_norm_lowmem(
                tf_eta,
                tf.constant(Corig[inds], dtype=dtype),
                loss0,
                loss1,
                tf.constant(y[inds], dtype=dtype),
                eps,
                gamma_reg,
                dtype,
                max_float_pow
            ),
            var_list = [tf_eta]
        )

        if verbose and i % 10 == 0:
            print('Current eta: {}'.format(tf_eta.numpy()))

    eta = tf_eta.numpy()

    return eta 

# Not sure that this will be enough in the float64 case.  
# Barely able to run in the float32 case.
# I suppose we could get Pi as a float32 matrix.  That should actually 
# be okay. I don't know what the cast time will be like, but it should be do-able
# 
# No seatbelts.  Make sure every input argument has the same type/
@tf.function
def make_pi(
        C,
        loss0,
        loss1,
        ytf,
        eta,
        gamma_reg,
        #gpu='/GPU:1',
        dtype=tf.float32,
        max_float_pow = _max_float_pow
        ):

    #with tf.device(gpu):

    # don't need to be super careful with tensorflow functions here
    # because we only run this once per iteration
    temp = tf.scalar_mul(1/gamma_reg, 
            tf.transpose(
                tf.tensordot(ytf, loss1, axes=0) +\
                tf.tensordot(1-ytf, loss0, axes=0)
            ) -\
            eta * C
        )

    # normalize to avoid overflow in float
    t_max = tf.reduce_max(temp, axis=0)
    t_max = tf.multiply(
            t_max - max_float_pow,
            tf.cast(t_max > max_float_pow, dtype=dtype)
        )

    temp = tf.exp(temp - t_max)

    u = 1.0/( tf.cast(C.shape[0], dtype=dtype)*tf.reduce_sum(temp, 0) )

    return tf.multiply(temp, u) 

# not sure if this requires a copy of C or not. 
# I suppose we could always make it on GPU1 (since this will be running on
# that GPU anyways
# I don't think that there is any reason to make pi by column, since we
# always need to store the full thing on disk at some point.
#def make_pi_bycol(
#        Cdata,
#        loss0,
#        loss1,
#        eta,
#        gamma_reg,
#        n,
#        #gpu='/GPU:1',
#        dtype=tf.float64
#        ):
#
#    for col, y in Cdata:
#        temp = tf.exp(
#                tf.scalar_mul(y, loss1) +\
#                tf.scalar_mul(tf.constant(1.0, dtype=dtype) -y, loss0) -\
#                tf.scalar_mul(eta/gamma_reg, col)
#            )
#
#        ucol = tf.math.reciprocal(
#                tf.scalar_mul( n, tf.reduce_sum(temp) )
#            )
#
#        res += ucol * tf.reduce_sum(
#                tf.multiply(col, temp)
#            )
#
#    return res

def sinkhorn_sgd_pi_2gpu(
        tf_eta,
        Corig, 
        y,
        C,
        loss0,
        loss1,
        eps,
        gamma_reg,
        epoch, 
        batch_size, 
        n = None,
        dtype=tf.float64,
        verbose = True,
        lr=0.05,
        momentum=0.9,
        init=0.1,
        newSoln=True,
        max_float_pow = _max_float_pow
        ):

    if newSoln: tf_eta.assign(init)

    if(verbose):
        print("***Attack step: finding root eta.", flush=True)
        t_s = time.time()

    eta = sinkhorn_sgd_norm_lowmem(
        tf_eta,
        Corig, 
        loss0,
        loss1,
        y,
        eps,
        gamma_reg,
        epoch, 
        batch_size, 
        n = n,
        dtype=dtype,
        verbose = verbose,
        lr=lr,
        momentum=momentum,
        max_float_pow=max_float_pow
        )


    if (verbose):
        print('Took %f' % (time.time() - t_s))
        print("discovered value of eta: {}".format(eta))

        print("***Attack step: constructing pi on GPU:1.", flush=True)
        t_s = time.time()

    with tf.device('/GPU:1'):
        pi = make_pi(
                C,
                loss0,
                loss1,
                tf.constant(y, dtype=dtype),
                tf.constant(eta, dtype=dtype),
                gamma_reg,
                dtype=dtype,
                max_float_pow =max_float_pow
                )
    if (verbose):
        print('Took %f' % (time.time() - t_s))

    return pi
