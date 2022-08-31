"""
Preliminary sampling of points
"""

import sys as sys
import numpy as np
from pyscf import gto
from pyscf import dft
from pyscf import hessian
from pyscf.hessian import thermo


def pre_sample(X0, nsample):
    """
    Pre-sampling of geometries and energies using provided
    normal modes + frequencies
    """

    # Build the PySCF Mole object
    mol         = gto.Mole()
    mol.verbose = 0
    mol.atom    = X0
    mol.basis   = '6-31g'
    mol.build()

    # Build and run the PySCF SCF object
    mf             = dft.RKS(mol)
    mf.grids.level = 4
    mf.grids.prune = True
    mf.xc          = 'b3lyp'
    mf.verbose     = 0
    mf.kernel()

    # Build and run the PySCF Hessian object
    n3   = mol.natm * 3
    hobj = hessian.rks.Hessian(mf)
    Hij  = hobj.kernel() #.transpose(0,2,1,3).reshape(n3,n3)

    # Get the normal modes and frequencies
    nm      = thermo.harmonic_analysis(mf.mol, Hij)
    freq    = nm['freq_au']
    n_modes = len(freq)
    modes   = nm['norm_mode'].reshape(n_modes,n3)

    # Frequency weight the normal modes
    sys.exit('\n remember to apply the frequency weighting...')
    
    # Sampling
    X = []
    E = []
    rand = np.random.standard_normal(nsample)
    for i in range(nsample):
        print(rand[i])
    
    return X, E
