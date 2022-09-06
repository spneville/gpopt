"""
Non-redundent internal coordinates class
"""

import sys as sys
import numpy as np
import chemcoord as cc

class Internals():
    def __init__(self, zmat0):
        """
        Internal class object constructor
        """

        # Guess geometry chemcoord Z-matrix object
        self.zmat0 = zmat0

        # Dimensions
        self.n_atom     = self.zmat0['bond'].to_numpy().size
        self.n_internal = self.count_internals()
        self.n_bond     = self.count_bonds()
        self.n_angle    = self.count_angles()
        self.n_dihedral = self.count_dihedrals()

        # Guess geometry internal coordinates
        self.intcoord0  = self.zmat_to_int(self.zmat0)

        # Guess geometry Cartesian coordinates
        self.xcoord0    = self.int_to_cart(self.intcoord0)
        
    def int_to_cart(self, intcoord):
        """
        Converts an array of internal coordinates
        to Cartesian coordinates
        """
        
        zmat = self.zmat0.copy()
        
        for i in range(self.n_bond):
            zmat.safe_loc[i+1, 'bond'] = intcoord[i]
        
        shift = self.n_bond
        for i in range(self.n_angle):
            zmat.safe_loc[i+2, 'angle'] = intcoord[i+shift]

        shift += self.n_angle
        for i in range(self.n_dihedral):
            zmat.safe_loc[i+3, 'dihedral'] = intcoord[i+shift]
            
        xcoord = self.zmat_to_cart(zmat)

        return xcoord

    def zmat_to_int(self, zmat):
        """
        Returns the array of internal coordinates
        given a chemcoord Z-matrix object
        """

        intcoord = np.zeros((self.n_internal), dtype=float)

        intcoord[0:self.n_bond] = \
            zmat['bond'].sort_index().to_numpy()[(self.n_atom-self.n_bond):]

        intcoord[self.n_bond:self.n_bond+self.n_angle] = \
            zmat['angle'].sort_index().to_numpy()[(self.n_atom-self.n_angle):] 
        
        intcoord[self.n_bond+self.n_angle:] = \
            zmat['dihedral'].sort_index().to_numpy()[(self.n_atom-self.n_dihedral):]

        return intcoord
    
    def zmat_to_cart(self, zmat):
        """
        Converts a chemcoord Z-matrix object to Cartesian
        coordinates (x1, y1, z1, ... , xn, yn, zn)
        """

        xcoord = np.zeros((3*self.n_atom), dtype=float)

        cart = zmat.get_cartesian()
        x    = cart['x'].to_numpy()
        y    = cart['y'].to_numpy()
        z    = cart['z'].to_numpy()

        for i in range(self.n_atom):
            xcoord[(i+1)*3-3] = x[i]
            xcoord[(i+1)*3-2] = y[i]
            xcoord[(i+1)*3-1] = z[i]

        return xcoord
        
    def count_internals(self):
        """
        Returns the number of internal coorindates
        """

        n_cart = 3*len(self.zmat0['bond'].tolist())
        
        axes = ['e_x', 'e_y', 'e_z']
        
        n_rt = 0

        n_rt += self.zmat0['b'].tolist().count('origin')

        for axis in axes:
            n_rt += self.zmat0['a'].tolist().count(axis)
            n_rt += self.zmat0['d'].tolist().count(axis)

        return  n_cart - n_rt

    def count_bonds(self):
        """
        Returns the number of bonds
        """
        
        n = len(self.zmat0['b'].tolist()) \
            - self.zmat0['b'].tolist().count('origin')
                
        return n
    
    def count_angles(self):
        """
        Returns the number of angles
        """

        axes = ['e_x', 'e_y', 'e_z']
        
        n = len(self.zmat0['a'].tolist())

        for axis in axes:
            n -= self.zmat0['a'].tolist().count(axis)

        return n

    def count_dihedrals(self):
        """
        Returns the number of dihedrals
        """

        axes = ['e_x', 'e_y', 'e_z']
        
        n = len(self.zmat0['d'].tolist())

        for axis in axes:
            n -= self.zmat0['d'].tolist().count(axis)

        return n
