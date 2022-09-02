"""
Module for the construction of normal modes
"""

import sys as sys
import numpy as np
import gpopt.core.constants as constants
import gpopt.coord.modes as modes
from pyscf import gto
from pyscf import dft
from pyscf import hessian
from pyscf.hessian import thermo

def modes_pyscf(geom0):
    """
    Construction of a Normal_modes object using a PySCF Hessian
    """
    
    # Build the PySCF Mole object
    mol         = gto.Mole()
    mol.verbose = 0
    mol.atom    = geom0
    mol.basis   = '6-31g'
    mol.build()

    # Build and run the PySCF SCF object
    mf    = dft.RKS(mol)
    mf.xc = 'b3lyp'
    E0    = mf.kernel()

    # Build and run the PySCF Hessian object
    ncoo = mol.natm * 3
    hobj = hessian.rks.Hessian(mf)
    Hij  = hobj.kernel()

    # Get the normal modes and frequencies
    nm_dict = thermo.harmonic_analysis(mf.mol, Hij)
    freq    = nm_dict['freq_au']
    freq_ev = freq * 5140.8096 * 1.23985e-4
    nmodes  = len(freq)
    qcoo    = nm_dict['norm_mode'].reshape(nmodes,ncoo).T

    # Masses
    names = [mol.atom[i][0] for i in range(mol.natm)]
    indx  = [constants.atom_name.index(name) for name in names]
    mass  = np.array([constants.atom_mass[i] for i in indx
                      for j in range(3)])

    # Mass-weight the normal modes
    qcoo = np.einsum('i,ij->ij', np.sqrt(mass), qcoo)

    # Create the Modes object
    x0 = np.array([geom0[i][1] for i in range(mol.natm)]).reshape(ncoo)
    modes_obj = modes.Modes(x0, ncoo, nmodes, mass, freq, qcoo)
    
    return modes_obj
