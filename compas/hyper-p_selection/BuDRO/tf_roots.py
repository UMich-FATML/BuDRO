#  On greatlakes, we have tensorflow 2.0
import tensorflow as tf
import numpy as np
from scipy.optimize import root_scalar
_max_float32_pow = 65.
_max_float64_pow = 709.

max_float_pow = _max_float32_pow

# All testing carried out with all matrices from np.random.uniform() (so not
# distance matrices or anything like that) (so not distance matrices or
# anything like that)

# These functions are working.  Testing shows that inner_full is ~10 times
# faster than inner_bycol when n is about 1000 and all of the inputs declared
# as constants.

# Only get 3-4 times speedup when n is 5000, but it takes longer to construct
# the inputs as well (need to allocate memory for L, V).  This does lead to faster
# solution time on sentiment... but might just be too much memory for adult.
# We will see

# When n = 10000, takes ~8-10 seconds to make L from loss0 and loss1, but each
# eval of inner_full takes ~0.5 seconds.  Each eval of inner_bycol takes ~1.5
# seconds.  So it depends on the number of function evaulations.  In general,
# we will probably have more than 10 evaluations, so we save time with the full
# method.

# For n = 30000, it takes ~25 seconds to make L from loss0 and loss1.  Inner
# full fails - it requires sending too much memory to a GPU.  Running on 2 GPUs
# doesn't get me the memory from both (still using too much memory).  Running
# with lowmem doesn't help - we still allocate the temp variable on the GPU.
# So this probably just won't work in general. (We're definitely planning to
# work with data sets larger than this).  Like... I can get it working, but
# this just will generally fail as the size of the data set gets larger.
#
# About 30 seconds to compute V from L and C on 4 cpus.  Not sure if we could
# combine the consruction of L in the same step (or still a way to skip the
# consturction of L).  And then 21 seconds/iteration for inner_full_lowmem.
# 3.6 seconds/iteration if we have temp and C already as tf.constants.  It
# takes ~23 seconds/iteration when C is a tf.constant and temp is not.  It
# takes ~18 seconds/iteration to make temp into a tf.constant. That needs to be
# done for every value of eta (every function evaluation).  No way to avoid the
# computation costs and the memory usage is just too high.
#
# Inner_bycol takes about 10 seconds per iteration (13/iteration on 2 gpus).
# 30 function evaluations = 6.5 minutes.  Not great.

# Using float32 (which is probably enough for adult...) 
# Inner_full: ~2.6 seconds/iteration (remember add ~25 seconds for making L)
# Inner_bycol:  ~6.6 seconds/iteration.
# We save if we have more than 6 function evaluations.  We will probably have
# more than 6. 
# Total timing: (25 + 30*2.6 = 1.7 minutes or 30*6.6 = 3.3 minutes)
# That's using bisection for 10**(-10) accuracy.
# (In practice, we often see only 10-15 iterations here - 
# try with real adult data set)
# That's honestly amazing
# Faster construction of L using np.kron.  Makes sense.
# Wonder if I can do that with GPU computations.
# Oh yeah. Takes 0.04 seconds to construct L like this.  
#   yeah GPUs are cool.  Holy crow that's fast.
# Oh my goodness.  Can do the dual even faster????
# Yep.  The dual takes less than one second to solve.  Wow.
# Dual takes ~0.75 seconds to solve.  That's like 22 seconds per boosting step.

# Function to do things column by column - avoid 
# passing the full matrix C to the GPU
#
# Low memory - so should be able to run with float64.  But evern keeping C and 
# Pi in memory is probably too much
@tf.function
def inner_bycol(eta, loss0, loss1, Cdata, gamma_reg, n, dtype=tf.float64):
    
    res = tf.constant(0.0, dtype=dtype)

    for col, y in Cdata:
        temp = tf.exp(
                tf.scalar_mul(y, loss1) +\
                tf.scalar_mul(tf.constant(1.0, dtype=dtype) -y, loss0) -\
                tf.scalar_mul(eta/gamma_reg, col)
            )

        ucol = tf.math.reciprocal(
                tf.scalar_mul( n, tf.reduce_sum(temp) )
            )

        res += ucol * tf.reduce_sum(
                tf.multiply(col, temp)
            )

    return res

# This is faster than bycol
@tf.function
def inner_full(eta, V, C, C0, dtype=tf.float64):

    temp = tf.exp( V - tf.multiply(eta, C0) )

    inner = tf.tensordot( 
        tf.reduce_sum(tf.multiply(C, temp), 0),
        tf.math.reciprocal( 
            tf.multiply(
                tf.cast(C.shape[0], dtype), 
                tf.reduce_sum(temp, 0) 
            ) 
        ),
        axes=1
    )

    return inner

# This is the fastest for n=10000... perhaps because we reduce communication
# costs?
#
# On full adult: this takes ~1.7 seconds per iteration (when we construct V on
# a second gpu before using this).  I think this is probably just due to
# increased communication costs here... That would be my guess anyways.  

# V is stored on the other GPU, so needs to be passed over every time.  But I
# also think that there is no other way to do this.

# Yep - just defining V again and then running inner_full_mul results in a OOM
# error.
#
#  No idea why by putting V on GPU0 and running the function on GPU 1 is faster
#  (~1.3 seconds per iteration).  For this example, I computed V first and C
#  was on CPU. In any case, this is not faster than the below for significantly
#  more difficulty working with memory.
#
# Dec 23: added normalization
@tf.function
def inner_full_mul(eta, V, C, gamma_reg, dtype=tf.float64, max_float_pow=65.):

    temp = V - tf.scalar_mul(tf.cast(eta/gamma_reg, dtype=dtype), C)

    # normalize to avoid overflow in float
    t_max = tf.reduce_max(temp, axis=0)
    t_max = tf.multiply(
            t_max - max_float_pow,
            tf.cast(t_max > max_float_pow, dtype=dtype)
        )

    temp = tf.exp(temp - t_max)

    inner = tf.tensordot( 
        tf.reduce_sum(tf.multiply(C, temp), 0),
        tf.math.reciprocal( 
            tf.multiply(
                tf.cast(C.shape[0], dtype), 
                tf.reduce_sum(temp, 0) 
            ) 
        ),
        axes=1
    )

    return inner

# This is the slowest of the inner_full functions though, for n=10000.
# Communications costs not so bad at this size.  But we still need to compute L
# every time we run this so... computation costs are definitely increased.  
#
# On the full adult data set, this takes ~1.3 seconds per iteration, resulting
# in around 36 seconds per sinkhorn evaluation (but not finding pi).
#
# Once we normalize, to allow for working with float32, it takes ~1.35-1.4
# seconds per function evaluation

# How to give an input signature - might allow for nicer/lower memory graph
# tracing, but you really want the implementation in the same file as your data
# set.
#(input_signature=[
#    tf.TensorSpec(tf.TensorShape([]), tf.float32), 
#    tf.TensorSpec(tf.TensorShape([36177]), tf.float32),
#    tf.TensorSpec(tf.TensorShape([36177]), tf.float32),
#    tf.TensorSpec(tf.TensorShape([36177]), tf.float32),
#    tf.TensorSpec(tf.TensorShape([36177, 36177]), tf.float32),
#    tf.TensorSpec(tf.TensorShape([]), tf.float32) 
#    ])

# I'm getting rid of the casting.  Be careful what data types you are working
# with

@tf.function
def inner_full_L(eta, loss0, loss1, ytf, C, gamma_reg, dtype=tf.float32):
    print("inner_full_L: TRACING GRAPH")

    temp = tf.scalar_mul(
            tf.math.reciprocal(gamma_reg),
            tf.transpose(
                tf.tensordot(ytf, loss1, axes=0) +\
                tf.tensordot(1-ytf, loss0, axes=0)
            ) -\
            tf.scalar_mul(eta, C)
        )

    # normalize to avoid overflow in float
    t_max = tf.reduce_max(temp, axis=0)
    t_max = tf.multiply(
            t_max - max_float_pow,
            tf.cast(t_max > max_float_pow, dtype=dtype)
        )

    temp = tf.exp(temp - t_max)

    inner = tf.tensordot( 
        tf.reduce_sum(tf.multiply(C, temp), 0),
        tf.math.reciprocal( 
            tf.scalar_mul(
                tf.cast(C.shape[0], dtype), 
                tf.reduce_sum(temp, 0) 
            ) 
        ),
        axes=1
    )

    return inner

# Construct temp separately and send it in - good for working with
# larger data sets, but slower
@tf.function
def inner_full_lowmem(eta, temp, C):

    inner = tf.tensordot( 
        tf.reduce_sum(tf.multiply(C, temp), 0),
        tf.math.reciprocal( 
            tf.multiply(
                tf.cast(C.shape[0], tf.float64), 
                tf.reduce_sum(temp, 0) 
            ) 
        ),
        axes=1
    )

    return inner

# Specifically construct L separately - this is not a tf.function 
# and send it in.
def inner_lowgpu(eta, L, C, gamma_reg):
    
    temp = np.exp( L/gamma_reg - eta*C/gamma_reg )
    return inner_full_lowmem(eta, temp, C)

# might need to normalize L as well as C to use float32.
# also maybe make more parts tf.functions for speed
#
# Will realistically only work will small examples
# we don't do any normalization.
def sinkhorn_tf(
        C,
        loss0,
        loss1,
        ytf,
        eps,
        gamma_reg,
        lowg=0.0,
        highg=5.0,
        dtype=tf.float64,
        max_float_pow=65.,
        verify=False,
        verbose=False,
        ):

    # quick construction of L (V) using tensorflow
    V =  tf.scalar_mul(
            tf.math.reciprocal(tf.cast(gamma_reg, dtype=dtype)),
            tf.transpose(
                tf.tensordot(tf.cast(ytf, dtype=dtype), loss1, axes=0) +\
                tf.tensordot(tf.cast(1-ytf, dtype=dtype), loss0, axes=0)
            )
        )

    if verify:
        obj0 = (tf.reduce_sum(tf.multiply(ytf, loss1)) +\
                tf.reduce_sum(tf.multiply(1-ytf, loss0)))/C.shape[0] -\
                gamma_reg * tf.cast(tf.math.log(1/C.shape[0]), dtype=dtype)
        if verbose:
            print("Original loss: {}".format(obj0.numpy()), flush=True)

    def obj_f(x):
        return tf.cast(eps, dtype) -\
                inner_full_mul(
                        tf.cast(x, dtype),
                        V,
                        C,
                        gamma_reg,
                        dtype,
                        max_float_pow
                    )

    # maybe want to use secant here if you can find some decent guesses
    sol = root_scalar(obj_f, bracket=[lowg,highg])
    eta = sol.root

    temp = V - tf.scalar_mul(tf.cast(eta/gamma_reg, dtype=dtype), C)

    # normalize to avoid overflow in float
    t_max = tf.reduce_max(temp, axis=0)
    t_max = tf.multiply(
            t_max - max_float_pow,
            tf.cast(t_max > max_float_pow, dtype=dtype)
        )

    temp = tf.exp(temp - t_max)

    u = tf.math.reciprocal( 
        tf.scalar_mul(
            tf.cast(C.shape[0], dtype), 
            tf.reduce_sum(temp, 0) 
        ) 
    )
    
    temp = tf.multiply(temp, u)
    if verify:
        # compute entropy
        pin = tf.boolean_mask(temp, temp>0)

        objf = tf.reduce_sum(tf.multiply(temp, gamma_reg*V)) -\
                gamma_reg*tf.reduce_sum(tf.multiply(pin, tf.math.log(pin)))
        print("Final value of inner max: {}".format(objf.numpy()), flush=True)

        diff = objf-obj0
        if verbose:
            print("Change in max: {}".format( diff.numpy() ))

        if diff.numpy() < 0:
            print("Warning: sinkhorn attack failed to increase loss objective.", 
                    flush=True)

    return temp, sol 

# Some notes about rounding the output pi:
# We find the root eta with precsion around 10**-9 or 10**-10
# No way does precision get better than that in our following computations
# Thus, setting entires smaller than that to 0 should basically be safe.

# In fact, we might just not have enough numerical stability here.

## Wonder if I could even save memory by combining these into two steps 
## (saves communication time - passing to the GPU from the CPU - and maybe
## even in the graph construction cause I can reduse repeated nodes?

# Using inner_full_L so we can work with the full adult data set
# This will have inner_full_L working all on one GPU
#
# All of the inputs should be floats.
def find_eta(
        C,
        loss0,
        loss1,
        ytf,
        eps,
        gamma_reg,
        lowg=0.0,
        highg=5.0,
        dtype=tf.float64,
        ):


    def obj_f(x):
        return tf.cast(eps, dtype) -\
                inner_full_L(
                        tf.constant(x, dtype),
                        loss0,
                        loss1,
                        ytf,
                        C,
                        gamma_reg,
                        dtype
                    )

    # maybe want to use secant here if you can find some decent guesses. Could
    # maybe use some different methods - like secant - if you feel adventurous
    sol = root_scalar(obj_f, bracket=[lowg,highg])

    return sol

# make pi using a different GPU to avoid OOM errors but also keep the SPEED
# This is not type safe (for the sake of memory, removing the casts seems to
# help quite a bit).  So be careful what you input.  For some reason, casting
# an existing constant tensor seems to require a lot of memory (even if it is
# something pretty small, like y, gamma_reg, or eps).  So INPUT THE ORIGINAL y
# IN TO THIS!  NOT A TENSORFLOW OBJECT!
@tf.function
def make_pi(
        C,
        loss0,
        loss1,
        ytf,
        eta,
        gamma_reg,
        #gpu='/GPU:1',
        dtype=tf.float32
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

    # normalize to avoid overflow in float32
    t_max = tf.reduce_max(temp, axis=0)
    offset = _max_float32_pow * tf.cast( t_max > _max_float32_pow, dtype=dtype)
    t_max -= offset

    temp = tf.exp(temp - t_max)

    u = 1.0/( tf.cast(C.shape[0], dtype=dtype)*tf.reduce_sum(temp, 0) )

    return tf.multiply(temp, u) 
