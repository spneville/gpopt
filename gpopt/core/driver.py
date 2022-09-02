"""
Optimisation driver: handles the preliminary sampling and calls to PySCF
and the optimiser
"""
import sys as sys
import numpy as np
import sklearn as sklearn
import gpopt.sampling.sampling as sampling
import gpopt.gpr.gpr as gpr
import gpopt.coord.hessian as hessian

class Driver:
    def __init__(self, geom):
        """
        Driver class object constructor
        """

        # Initial geometry
        self.geom0      = geom
        
        # Training set {(X_i, E_i)}
        self.ntrain     = 0
        self.train_set  = None

        # Preliminary sampling
        self.nprelim    = 25
        self.norm_bound = 2.5
        
    def run(self):
        """
        Runs a geometry optimisation using a GPR surrogate
        potential constructed on-the-fly
        """

        # Initial, approximate normal modes
        mode_obj = hessian.modes_pyscf(self.geom0)

        
        # Preliminary sampling of points
        Q, E = sampling.pre_sample(mode_obj, self.geom0,
                                   self.nprelim, self.norm_bound)

        # Construct the GPR object
        gpr_obj = gpr.gpr_scikit(Q, E)

        # Fit to the preliminary data
        gpr_obj.fit(Q, E)
        
        # Testing
        #Q_test, E_test = sampling.pre_sample(mode_obj, self.geom0,
        #                                     100, self.norm_bound,
        #                                     inc_Q0=False)
        #
        #mean_prediction, std_prediction = gpr_obj.predict(Q_test,
        #                                                  return_std=True)
        #
        #rmsd = np.sqrt(np.sum((mean_prediction - E_test)**2) / len(E_test))
        #
        #print('\n RMSD:', rmsd, ' eV')

        # Optimisation on the preliminary GPR potential
        
        return
