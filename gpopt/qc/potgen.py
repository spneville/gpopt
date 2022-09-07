"""
Potential generator module
"""
import sys as sys
import numpy as np
import gpopt.core.constants as constants
from pyscf import gto
from pyscf import dft


class Potgen():
    def __init__(self, basis, xc, atom_lbls, mode_obj, E0):
        """
        Potgen class object constructor
        """

        # Basis and XC functional
        self.basis = basis
        self.xc    = xc

        # Atom labels
        self.atm_lbls = atom_lbls

        # Normal modes object
        self.mode_obj = mode_obj

        # Reference energy
        self.E0 = E0
        
    def potfunc(self, Q):
        """
        Returns the potential at the point Q (in terms of normal modes)
        """

        # Cartesian coordinates
        X = self.mode_obj.q2x(Q)

        # PySCF calculation
        mol         = gto.Mole()
        mol.verbose = 0
        X3       = X.reshape(self.mode_obj.natm, 3)
        mol.atom = [[self.atm_lbls[i], list(X3[i])]
                    for i in range(self.mode_obj.natm)]
        mol.basis = self.basis
        mol.build()
        mf    = dft.RKS(mol)
        mf.xc = self.xc
        E     = mf.kernel()
        
        return (E - self.E0) * constants.eh2ev

    
class Potgen_int():
    def __init__(self, basis, xc, atom_lbls, int_obj, E0):
        """
        Potgen class object constructor for internal
        coordinates
        """

        # Basis and XC functional
        self.basis = basis
        self.xc    = xc

        # Atom labels
        self.atm_lbls = atom_lbls

        # Internal coordinates object
        self.int_obj = int_obj

        # Reference energy
        self.E0 = E0
        
    def potfunc(self, Q):
        """
        Returns the potential at the point Q (in terms of normal modes)
        """

        # Cartesian coordinates
        X = self.int_obj.int_to_cart(Q)

        # PySCF calculation
        mol         = gto.Mole()
        mol.verbose = 0
        X3       = X.reshape(self.int_obj.n_atom, 3)
        mol.atom = [[self.atm_lbls[i], list(X3[i])]
                    for i in range(self.int_obj.n_atom)]
        mol.basis = self.basis
        mol.build()
        mf    = dft.RKS(mol)
        mf.xc = self.xc
        E     = mf.kernel()
        
        return (E - self.E0) * constants.eh2ev

    
