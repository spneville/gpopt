"""
Optimisation driver: handles the preliminary sampling and calls to PySCF
and the optimiser
"""
import sys as sys
import numpy as np

class Driver:
    def __init__(self, geom):
        """
        Driver class object constructor
        """

        # Input geometry
        atom_lbl = [geom[i][0] for i in range(len(geom))]
        X0       = []
        [X0.append(geom[i][1:][0][j]) for i in range(len(geom))
         for j in range(3)]
        X0       = np.array(X0)

        self.X0       = X0
        self.atom_lbl = atom_lbl

        # Training set {(X_i, E_i)}
        self.ntrain    = 0
        self.train_set = None
        
    
    def run(self):
        """
        Runs a geometry optimisation using a GPR surrogate
        potential constructed on-the-fly
        """

        # preliminary sampling of points
        sys.exit()
        
        return
