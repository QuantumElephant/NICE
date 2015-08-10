#!/usr/bin/env python

from __future__ import division

import numpy as np
from scipy.optimize import fsolve

class ExactEqmSolver(object):
    ''' Solves for final mol fractions, or concentrations through non-linear equations '''

    def __init__(self, initial_concentrations, keq_values, stoich_coeff, initial_guess = None):
        '''
        Arguments:
        ----------
        initial_concentrations: list
            A vector containing the initial concentrations of each species. Order of species in this 
            list should correspond to that of the stoich_coeff columns. In Molar (mol/L).
        keq_values: list
            The equillibrim constant values for each reaction. Order should correspond to that of 
            the rows in the stoich_coeff matrix.
        stoich_coeff: numpy array
            Each column represents a species number, while each row represents a reversible reaction.
            Coefficients are negative if the species is a reactant, positive if the species is a 
            product, and zero if the species doesn't participate in the reaction at all. Every species 
            must be part of each reaction row, even if it doesn't participate in the reaction (the 
            coefficient will be zero in that case.
        initial_guess: list
            Guess for the zeta values of each reaction, but *not* the initial concentrations.
        keq_mol_frac: bool
            When True, the equillibrium constant is expressed in terms of mol fractions. When false, the equillibrium 
            constant is expressed in terms of concentrations.
        '''
        
        self.initial_concentrations = initial_concentrations
        self.keq_values = keq_values
        self.stoich_coeff = stoich_coeff
        self.initial_guess = initial_guess
        
        self.nspecies = len(self.initial_concentrations)
        self.nreactions = len(self.keq_values)


    def setup_keq_expressions(self, z, **kwargs):
        
        mol_exps = []
        for species, concentration in enumerate(self.initial_concentrations):
            mol_exp = concentration
            for i, coeff in enumerate(np.nditer(self.stoich_coeff[:,species])):
                mol_exp = mol_exp + coeff*z[i]
            mol_exps.append(mol_exp)
        

        keq_exps = []
        for rxn, keq in enumerate(self.keq_values):
            keq_exp = 1
            for expr, coeff in zip(mol_exps, np.nditer(self.stoich_coeff[rxn,:])):
                term = expr**coeff
                keq_exp = keq_exp*term
            keq_exp = keq_exp - self.keq_values[rxn]
            keq_exps.append(keq_exp)

        if 'return_value' in kwargs:
            if kwargs['return_value'] == 'mol_exps':
                return mol_exps        
    
        return keq_exps
    

    # Use the exact jacobian
    def get_zeta_values(self):
        '''
        Determines the extents of reactions (zeta values) by solving a system of nonlinear equations.

        Returns:
        --------
        zeta_values: list
            The extents of reaction determined by solving the nonlinear keq equations.
        '''

        if self.initial_guess == None:
            self.initial_guess = [0]*self.nreactions # All zeros is usually a stationary point- guess strongly reccomended!
        zeta_values = fsolve(self.setup_keq_expressions, self.initial_guess)
        print "The calculated extents of reaction are %s" %(zeta_values)
       
        return zeta_values

    
    def solve_final_concentrations(self):
        '''
        Runs the class using a single method.
        '''

        zeta_values = self.get_zeta_values()
        final_concentrations = self.setup_keq_expressions(zeta_values, return_value = 'mol_exps')
            
        print 'The final concentrations are: %s' %(final_concentrations)

