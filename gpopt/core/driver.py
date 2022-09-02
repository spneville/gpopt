"""
Optimisation driver: handles the preliminary sampling and calls to PySCF
and the optimiser
"""
import sys as sys
import numpy as np
import sklearn.gaussian_process as gp
import gpopt.sampling.sampling as sampling
import gpopt.gpr.surrogate as surrogate
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
        self.nprelim    = 55
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

        # Characteristic lengths
        char_lengths = np.array([1.
                                 for i in range(mode_obj.nmodes)])
        
        # Construct the GPR object
        kernel = 1 * gp.kernels.RBF(length_scale=char_lengths,
                                    length_scale_bounds=(1e-2, 1e2))

        gaussian_process = gp.GaussianProcessRegressor(kernel=kernel,
                                                       n_restarts_optimizer=9)

        gaussian_process.fit(Q, E)
        
        print('\n', gaussian_process.kernel_)


        # Test
        Q_test, E_test = sampling.pre_sample(mode_obj, self.geom0,
                                             100, self.norm_bound,
                                             inc_Q0=False)

        mean_prediction, std_prediction = \
            gaussian_process.predict(Q_test, return_std=True)

        rmsd = np.sqrt(np.sum((mean_prediction - E_test)**2) / len(E_test))

        print('\n RMSD:', rmsd, ' eV')
        
        return
