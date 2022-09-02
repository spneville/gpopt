"""
Construction of GPR objects
"""

import sys as sys
import numpy as np
import sklearn.gaussian_process as gp

def gpr_scikit(Q, E):
    """
    Constructs a scikit-learn GaussianProcessRegressor
    object
    """

    char_lengths = np.array([1. for i in range(Q.shape[1])])
    
    kernel = 1 * gp.kernels.RBF(length_scale=char_lengths,
                                length_scale_bounds=(1e-2, 1e2))
    
    gpr_obj = gp.GaussianProcessRegressor(kernel=kernel,
                                          n_restarts_optimizer=9,
                                          normalize_y=True)

    return gpr_obj
