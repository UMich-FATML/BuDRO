
import numpy as np
from numpy.random import multivariate_normal as nrm

def line(m, b, x):
    return m*x + b

def simData(mean0, mean1, cov, size0=100, size1=25):
    
    sample0 = nrm(mean0, cov, size=size0)
    sample1 = nrm(mean1, cov, size=size1)

    slope0 = ( sample0[:,1].max() - sample0[:,1].min() ) / ( sample0[:,0].max() - sample0[:,0].min() )
    slope1 = ( sample1[:,1].min() - sample1[:,1].max() ) / ( sample1[:,0].max() - sample1[:,0].min() )

    b0 = slope0*(-mean0[0]) + mean0[1]
    b1 = slope1*(-mean1[0]) + mean1[1]

    y = np.array(
        [1 if pt[1] >= line(slope0, b0, pt[0]) else 0 for pt in sample0] +\
        [1 if pt[1] >= line(slope1, b1, pt[0]) else 0 for pt in sample1]
    )

    X = np.concatenate((sample0, sample1))

    return X, y
