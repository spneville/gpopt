"""
Optimisation driver: handles the preliminary sampling and calls to PySCF
and the optimiser
"""
import sys as sys
import numpy as np
import sklearn as sklearn
import skopt as skopt
import gpopt.sampling.sampling as sampling
import gpopt.gpr.gpr as gpr
import gpopt.coord.hessian as hessian
import gpopt.coord.internals as internals
import gpopt.qc.potgen as potgen
import gpopt.core.constants as constants
import chemcoord as cc

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
        self.n_prelim   = 25
        self.norm_bound = 0.75

        # Bayesian optimisation variables
        self.n_calls    = 75
        
        # Reference energy
        self.E0         = None
        
    def run(self, xyz_file):
        """
        Runs a geometry optimisation using a GPR surrogate
        potential constructed on-the-fly
        """

        # guess geometry Z-matrix object
        zmat0 = cc.Cartesian.read_xyz(xyz_file).get_zmat()

        # internal coordinate object
        int_obj = internals.Internals(zmat0)

        # Preliminary sampling of points
        #Q, E, E0 = sampling.pre_sample(mode_obj, self.geom0,
        #                               self.n_prelim, self.norm_bound)
        #self.E0  = E0


        sys.exit()
        
        # Potential generator object
        atom_lbls = [self.geom0[i][0] for i in range(len(self.geom0))]
        potgen_obj = potgen.Potgen('cc-pvdz', 'b3lyp', atom_lbls, mode_obj,
                                   self.E0)

        # Bayesian optimisation
        x0 = np.ndarray.tolist(Q)
        y0 = np.ndarray.tolist(E)
        lower = -5.
        upper = 5.
        dimensions = [[lower, upper] for i in range(mode_obj.nmodes)]
        res = skopt.gp_minimize(potgen_obj.potfunc,
                                dimensions,
                                acq_func='gp_hedge',
                                n_calls=self.n_calls,
                                random_state=999,
                                noise=1e-4,
                                verbose=True,
                                x0=x0,
                                y0=y0)

        print(2*'\n',
              mode_obj.q2x(np.array(res.x)).reshape(mode_obj.natm,3))
        print(2*'\n', potgen_obj.potfunc(np.array(res.x))
              / constants.eh2ev + self.E0)

        sys.exit()
        
        
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
