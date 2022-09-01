"""
Preliminary sampling of points
"""

import sys as sys
import numpy as np
import gpopt.core.constants as constants
import gpopt.core.modes as modes
from pyscf import gto
from pyscf import dft
from pyscf import hessian
from pyscf.hessian import thermo


def pre_sample(geom0, nsample):
    """
    Pre-sampling of geometries and energies using provided
    normal modes + frequencies
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

    # Coordinates: uniformly sampled normal mode vectors with lengths uniformly
    #              drawn from [0,a]
    X = []
    a = 2.
    length = np.random.uniform(0., a, (nsample))
    qpt    = np.random.uniform(0., 1., (nsample, modes_obj.nmodes))    
    qpt_norm = np.array([qpt[i] * (length[i] / np.linalg.norm(qpt[i]))
                         for i in range(nsample)])
    for i in range(nsample):
        X.append(modes_obj.q2x(qpt_norm[i]))
    X = np.array(X)
        
    # Energies
    E = []
    for i in range(nsample):

        X3       = X[i].reshape(mol.natm, 3)
        mol.atom = [[names[i], list(X3[i,:])] for i in range(mol.natm)]
        mol.build()
        
        mf    = dft.RKS(mol)
        mf.xc = 'b3lyp'
        E.append(mf.kernel())

        print('Geom ', i, 'E (eV) = ', (E[i]-E0)*27.2114)
        
    E = np.array(E)
        
    return X, E
