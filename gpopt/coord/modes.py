"""
Normal mode class object
"""

import sys as sys
import numpy as np

class Modes():
    def __init__(self, x0, ncoo, nmodes, mass, freq, qcoo):
        """
        Modes class object constructor
        """

        # Cartesian coordinates (Angstrom)
        self.x0     = x0

        # Numer of atoms
        self.natm   = int(x0.size / 3)
        
        # Number of normal modes and Cartesian coordinates
        self.nmodes = nmodes
        self.ncoo   = ncoo

        # Masses (amu) and frequencies (ev)
        self.mass   = mass
        self.freq   = freq

        # Mass-weighted normal modes
        self.qcoo   = qcoo
        
        # Transformation matrices: between frequency-weighted
        # normal modes and Cartesian coordinates in Angstrom
        self.coonm, self.nmcoo = self.get_trans()
        

    def get_trans(self):
        """
        Constructs the Cartesian-to-normal mode
        transformation matrices
        """
        
        fac = np.array([15.4644*np.sqrt(np.abs(self.freq[i])*self.mass[j])
                        for i in range(self.nmodes)
                        for j in range(self.ncoo)])
        fac = fac.reshape(self.nmodes,self.ncoo)
        
        nmcoo = np.multiply((1./fac).T, self.qcoo)
        coonm = np.multiply(fac, self.qcoo.T)
        
        return coonm, nmcoo

    
    def q2x(self, q):
        """
        Transforms the normal mode coordinates q
        to Cartesians in Angstrom
        """
        return self.x0 + np.matmul(self.nmcoo, q)


    def x2q(self, x):
        """
        Transforms Cartesians x (in Angstrom) to
        normal mode coordinates
        """
        return np.matmul(self.coonm, x-self.x0)
