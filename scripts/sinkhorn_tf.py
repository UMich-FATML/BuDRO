import tensorflow as tf
import numpy as np
import copy

from scipy.optimize import root_scalar

# Finding the optimal transport map using entropic regularization without sgd
# this is a class for reusing memory and stuff like that.

# I want these values set up before any instantiations of
# the sinkhorn_attack class, so that I can get the proper
# sizes of the inputs to the tensorflow functions.
#
# That's why I'm doing this simple, silly looking "inheritance"
#
# The idea: you set up some parameters and then instantiate your
# functions with the fixed types.  
# We copy in the fixed types from the current sinkhorn parameters
# so that they remain fixed for a given attack instantiation
#
# Apologies to the user if this makes you cringe.  I am just a dirty hacker.
class tf_params():

    # This is tailored for full adult.
    n = 36177
    #n=150
    dtype = tf.float32

    # should make this a float.  Need to multiply exp(_max_float_pow)
    # by approximately n^2 and still not get overflow.
    # Max power is about 88 for float32
    # Max power is about 709 for float64
    # log(10**5) ~ 11.5
    max_float_pow = 65.0

    @classmethod
    def set_params(cls,C,dtype=tf.float32,max_pow=65.,sugg=False,verbose=True):
        cls.n = C.shape[0]
        cls.dtype = dtype

        sugg_pow = 65.
        if dtype is tf.float32:
            sugg_pow = np.floor(88 - 2*np.log(cls.n))
        elif dtype is tf.float64:
            sugg_pow = np.floor(709 - 2*np.log(cls.n))

        if verbose:
            print("Suggested value of max_pow: {}".format(sugg_pow))

        if sugg:
            cls.max_float_pow = max_pow
            if verbose: print("Using suggested value of max_float_pow")
        else:
            cls.max_float_pow = max_pow
            if verbose: print("Ignoring suggestion and" +\
                    " using input value of max_pow")

class sinkhorn():

    def __init__(self, Corig, y, dtype=tf.float32, verbose=True):

        self.C = tf.constant(Corig, dtype=dtype)
        self.ytf = tf.constant(y, dtype=dtype)
    
        self.n = Corig.shape[0]
        self.dtype = dtype

        self.verbose=verbose

        self.max_float_pow = copy.deepcopy(tf_params.max_float_pow)

        if self.n != tf_params.n or self.dtype != tf_params.dtype:
            print("Warning: inputs do not match " +\
                    "fixed tensorflow parameters.")
            print("Warning: sinkhorn functions will be created with " +\
                    "incorrect input_signatures")
            print("Warning: No guarantees about how graph tracing will " +\
                    "be accomplished.")
            print("Warning: But your code might still run")
            


        # attempt trace of inner_full_L
        if self.verbose: print("*** Attempting trace of innner")

        success=False
        for i in range(10):
            try:
                self.inner_full_L(
                    tf.constant(0.3, self.dtype),
                    tf.constant(np.zeros(self.n), self.dtype),
                    tf.constant(np.zeros(self.n), self.dtype),
                    tf.constant(0.005, self.dtype),
                )

                success=True
            except:
                print("Trace attempt {} failed.".format(i))

                # delete the cache of graphs in the function to force a re-trace
                # from stackoverflow
                # who knows if this will work in future versions of tensorflow...
                (self.inner_full_L
                        ._stateful_fn
                        ._function_cache
                        ._garbage_collectors[0]
                        ._cache
                        .clear()
                )

            if success:
                if self.verbose: print("Trace attempt {} worked!".format(i))
                break

        if not success and self.verbose:
            print("Error: unable to trace graph of inner_full_L")
            print("Error: expect bad things to occur")

    
    # INNER_FULL_L
    # Define inner_full_L here so that we can use the data in the spec
    # and the data in the definition of the function
    #
    # This is consistently taking 1.4-1.5 seconds per evaulation,
    # with an average of around 1.43 or 1.44
    @tf.function(input_signature=[
        tf.TensorSpec(tf.TensorShape([]), tf_params.dtype), 
        tf.TensorSpec(tf.TensorShape([tf_params.n]), tf_params.dtype),
        tf.TensorSpec(tf.TensorShape([tf_params.n]), tf_params.dtype),
        tf.TensorSpec(tf.TensorShape([]), tf_params.dtype) 
        ])
    def inner_full_L(self, eta, loss0, loss1, gamma_reg):
        if self.verbose: print("inner_full_L: TRACING GRAPH")

        temp = tf.scalar_mul(
                tf.math.reciprocal(gamma_reg),
                tf.transpose(
                    tf.tensordot(self.ytf, loss1, axes=0) +\
                    tf.tensordot(1-self.ytf, loss0, axes=0)
                ) -\
                tf.scalar_mul(eta, self.C)
            )

        # normalize to avoid overflow in float32
        t_max = tf.reduce_max(temp, axis=0)
        offset = tf.scalar_mul(
                self.max_float_pow,
                tf.cast(t_max > self.max_float_pow, dtype=self.dtype)
            )
        t_max -= offset

        temp = tf.exp(temp - t_max)

        inner = tf.tensordot( 
            tf.reduce_sum(tf.multiply(self.C, temp), 0),
            tf.math.reciprocal( 
                tf.scalar_mul(
                    tf.constant(self.n, dtype=self.dtype), 
                    tf.reduce_sum(temp, 0) 
                ) 
            ),
            axes=1
        )

        return inner

    #### FIND_ETA
    def find_eta(
            self,
            loss0,
            loss1,
            eps,
            gamma_reg,
            lowg=0.0,
            highg=4.0,
            ):

        eps = tf.cast(eps, self.dtype)

        # making this a tf.function leads to out-of-memory errors
        # since we cannot control the type of the input.
        def obj_f(x):
            return eps -\
                    self.inner_full_L(
                            tf.constant(x, self.dtype),
                            loss0,
                            loss1,
                            gamma_reg
                        )

        # maybe want to use secant here if you can find some decent guesses. Could
        # maybe use some different methods - like secant - if you feel adventurous
        self.sol = root_scalar(obj_f, bracket=[lowg,highg])

        return self.sol

    @tf.function
    def make_pi(
            self,
            loss0,
            loss1,
            eta,
            gamma_reg,
            #gpu='/GPU:1',
            ):

        #with tf.device(gpu):

        # don't need to be super careful with tensorflow functions here
        # because we only run this once per iteration
        temp = tf.scalar_mul(1/gamma_reg, 
                tf.transpose(
                    tf.tensordot(self.ytf, loss1, axes=0) +\
                    tf.tensordot(1-self.ytf, loss0, axes=0)
                ) -\
                eta * self.C
            )

        # normalize to avoid overflow in float32
        # this works but doesn't do exactly what I want - don't want 
        # to normalize rows that are smaller than max_float_pow
        t_max = tf.reduce_max(temp, axis=0)
        offset = self.max_float_pow * tf.cast(
                t_max > self.max_float_pow, dtype=self.dtype
                )
        t_max -= offset

        temp = tf.exp(temp - t_max)

        u = 1.0/( tf.constant(self.n, self.dtype)*tf.reduce_sum(temp, 0) )

        return tf.multiply(temp, u) 
