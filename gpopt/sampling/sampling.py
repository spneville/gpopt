"""
Preliminary sampling of points
"""

import sys as sys
import numpy as np
import gpopt.core.constants as constants
from pyscf import gto
from pyscf import dft


def pre_sample(modes_obj, geom0, nsample, norm_bound, inc_Q0=True):
    """
    Pre-sampling of geometries and energies using provided
    normal modes + frequencies
    """

    # Number of displaced geometries
    if (inc_Q0):
        ndisp = nsample -1
    else:
        ndisp = nsample
    
    # Atom labels
    names = [geom0[i][0] for i in range(modes_obj.natm)]
    
    # Coordinates: uniformly sampled normal mode vectors with lengths
    #              uniformly drawn from [0, norm_bound]
    X = []
    Q = []
    length = np.random.uniform(0., norm_bound, (nsample))
    qpt    = np.random.uniform(-1., 1., (nsample, modes_obj.nmodes))    
    qpt_norm = np.array([qpt[i] * (length[i] / np.linalg.norm(qpt[i]))
                         for i in range(nsample)])

    if (inc_Q0):
        Q.append(np.zeros((modes_obj.nmodes), dtype=float))

    for i in range(ndisp):
        X.append(modes_obj.q2x(qpt_norm[i]))
        Q.append(qpt_norm[i])
    X = np.array(X)
    Q = np.array(Q)
    
    # Energies
    E = []
    mol         = gto.Mole()
    mol.verbose = 0
    mol.atom    = geom0
    mol.basis   = 'cc-pvdz'
    mol.build()
    mf    = dft.RKS(mol)
    mf.xc = 'b3lyp'
    E0    = mf.kernel()

    if (inc_Q0):
        E.append(E0)

    for i in range(ndisp):

        X3       = X[i].reshape(modes_obj.natm, 3)
        mol.atom = [[names[i], list(X3[i,:])]
                    for i in range(modes_obj.natm)]
        mol.build()
        
        mf    = dft.RKS(mol)
        mf.xc = 'b3lyp'
        E.append(mf.kernel())

        print('Geom ', i+1, 'E (eV) = ', (E[-1]-E0) * constants.eh2ev)
        
    E = np.array(E)
    
    return Q, (E - E0)*constants.eh2ev, E0
