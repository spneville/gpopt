"""
Optimisation driver: handles the preliminary sampling and calls to PySCF
and the optimiser
"""
import sys as sys
import numpy as np
import skopt as skopt
import gpopt.sampling.sampling as sampling
import gpopt.coord.internals as internals
import gpopt.qc.potgen as potgen
import gpopt.core.constants as constants
import chemcoord as cc

class Driver:
    def __init__(self, geom):
        """
        Optimisation driver class object constructor
        Probably not really necessary to have this as
        a class, but will do for now
        """

        # Initial geometry
        self.geom0      = geom
        
        # Training set {(X_i, E_i)}
        self.ntrain     = 0
        self.train_set  = None

        # Preliminary sampling
        self.n_prelim   = 2

        # Bayesian optimisation variables
        self.n_calls    = 100
        
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
        Q, E, self.E0 = sampling.pre_sample_int(int_obj, self.geom0,
                                                self.n_prelim)
        
        # Potential generator object
        atom_lbls  = [self.geom0[i][0] for i in range(len(self.geom0))]
        potgen_obj = potgen.Potgen_int('cc-pvdz', 'b3lyp', atom_lbls,
                                       int_obj, self.E0)

        # Bayesian optimisation
        dimensions = np.ndarray.tolist(int_obj.get_bounds())
        x0 = np.ndarray.tolist(Q)
        y0 = np.ndarray.tolist(E)
        n_calls = self.n_calls + self.n_prelim*2*int_obj.n_internal + 1
        res = skopt.gp_minimize(potgen_obj.potfunc,
                                dimensions,
                                acq_func='gp_hedge',
                                n_calls=n_calls,
                                n_initial_points=0,
                                n_random_starts=0,
                                random_state=999,
                                noise=1e-6,
                                verbose=True,
                                x0=x0,
                                y0=y0)

        # Output the optimised geometry and energy
        print(2*'\n',
              int_obj.int_to_cart(np.array(res.x)).reshape(int_obj.n_atom,3))
        print(2*'\n', potgen_obj.potfunc(np.array(res.x))
              / constants.eh2ev + self.E0)

        return
