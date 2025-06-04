# 11 June 2020
#
# Here is an example of running simulated data using the 
# sinkhorn_tf method (with BuDRO) following the setup in the 
# Appendix of the paper.
#
# File names (for saving the data) appear at the end of the 
# uncommented part of this file
#
# Run this with `python sim_tf_test.py`
# Plot the data in the notebook `simulated-plots.ipynb`


import tensorflow as tf
import numpy as np
import pickle
import time

import sys
sys.path.append('../scripts')

import make_data
import tf_xgboost_log as txl


_dtype=tf.float32
_n_iter=150

eps = 0.1
gamma_reg = 0.03

mean0 = np.array([-2,0])
mean1 = np.array([2,0])
cov = np.array([[0.2, 0],[0, 5]])

print("Making simulated data")
X,y = make_data.simData(mean0, mean1, cov, size0=125, size1=25)

bluepts = X[y==1]
redpts = X[y==0]

blueInds = np.where(y==1)[0]
redInds = np.where(y==0)[0]

print("Computing distances")
# Fair metric - y distance only
def dist(x1, x2):
    return np.abs(x1[1] - x2[1])

Corig = np.array([ [ dist(xi, xj) for xi in X] for xj in X])

# xgboost parameters used in original file
param = {'max_depth':2, 'eta':0.1, 'objective':'binary:logistic', 'min_child_weight':3/X.shape[0], 'lambda':0.2}

# boost with timing
t_s = time.time()
bst, pi = txl.boost_sinkhorn_tf(
        X, y, 
        tf.constant(Corig, dtype=_dtype), 
        _n_iter, 
        pred=None,  # we don't have a first guess
        eps=eps, 
        gamma_reg=gamma_reg,
        param=param,
        bst=None,
        lowg=0.0,
        highg=40.0,
        verbose=False,
        verify=False,
        roottol=10**(-16),
        dtype=_dtype,
        max_float_pow = 65.,
        outfunc=None
)

print('Took %f' % (time.time() - t_s))

# output some summary imformation
final = (Corig*pi.numpy()).sum()
print("Constraint value: {}".format(final))

pickle.dump(bst, open('sim_tf.pickle.dat', 'wb'))
np.savez("sim_tf_run.npz", X=X, y=y, pi=pi)




'''
# Below is very general code that can be used on LARGE data sets.
# It should essentially work on adult, for example.  It requires that 
# two GPUs are in the compute environment, however, which is way 
# overkill for this simple example.
# Nonetheless, it is good to see this general example.

# this would replace everything under the definition of param
# (line 42) if you wanted to use this more powerful framework

from sinkhorn_tf import tf_params

# Create sinkhorn attack paramters and data structures
print("Setting sinkhorn paramters")
params = tf_params()
params.set_params(Corig, dtype=_dtype, max_pow=65.0, sugg=True)

print("Parameters: n = {}".format(tf_params.n))

from sinkhorn_tf import sinkhorn

sink = sinkhorn(Corig, y, _dtype, verbose=True)

# boost with timing
t_s = time.time()
bst, pi = txl.boost_sinkhorn_2gpu(
        X,
        y,
        sink,
        _n_iter,
        pred=None,
        eps=eps,
        gamma_reg=gamma_reg,
        param=param,
        lowg=0.0,
        highg=40.0,
        verbose=False,
        dtype=tf.float32
        )

print('Took %f' % (time.time() - t_s))

# output some summary imformation
final = (Corig*pi.numpy()).sum()
print("Constraint value: {}".format(final))

pickle.dump(bst, open('sim_tf.pickle.dat0', 'wb'))
np.savez("sim_tf_run0.npz", X=X, y=y, pi=pi)
'''
