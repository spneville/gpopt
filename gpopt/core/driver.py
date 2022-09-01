"""
Optimisation driver: handles the preliminary sampling and calls to PySCF
and the optimiser
"""
import sys as sys
import numpy as np
import gpopt.sampling.sampling as sampling

class Driver:
    def __init__(self, geom):
        """
        Driver class object constructor
        """

        # Initial geometry
        self.geom0 = geom
        
        # Training set {(X_i, E_i)}
        self.ntrain    = 0
        self.train_set = None

        # Preliminary sampling
        self.nprelim = 25
        
    def run(self):
        """
        Runs a geometry optimisation using a GPR surrogate
        potential constructed on-the-fly
        """

        # Preliminary sampling of points
        X, E = sampling.pre_sample(self.geom0, self.nprelim)

        # Construct the GPR object
        
        return
