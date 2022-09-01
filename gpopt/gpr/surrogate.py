"""
GPR surrogate potential class object
"""

import sys as sys
import numpy as np

class Surrogate():
    def __init__(self, mode_obj, X_init, E_init, char_lengths, mindx=2):
        """
        Surrogate potential class object constructor
        """

        # quasi-normal mode object
        self.mode_obj     = mode_obj
        
        # Training set
        self.ntrain       = np.shape(E_init)
        self.X            = X_init
        self.E            = E_init

        # Characteristic length scales
        self.char_lengths = char_lengths

        # Matern covariance function index
        self.matern_indx  = mindx


    #def add_points(self, X, E):
    #    """
    #    Adds points to the training set
    #    """
    #    
    #    return
