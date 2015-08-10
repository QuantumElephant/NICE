#!/usr/bin/env python

from __future__ import division

import numpy as np
from scipy.optimize import fsolve

class ExactEqmSolver(object):
    '''Solves for final concentrations by solving a system of nonlinear equations'''

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
        '''
        
        self.initial_concentrations = initial_concentrations
        self.keq_values = keq_values
        self.stoich_coeff = stoich_coeff
        self.initial_guess = initial_guess
        

    def setup_keq_expressions(self, z, **kwargs):
        '''
        Creates the correct keq expressions in terms of the final concentrations for each species.

        Arguments:
        ----------
        z: list
            Each element represents a different zeta value.
        kwargs: dict
            Used to alter the return value of the function (to get the mol_exps back instead of keq_exps).
            The only acceptable keyword currently is 'return_value' and the corresponding value is 'mol_exp'.
            If no kwargs are specified, the function returns keq_exps by default.
        
        Returns:
        --------
        mol_exps: list
            Expressions for the final concentrations of each species in terms of the zeta for each reaction
            the species participates in.
        keq_exps: list
            Expressions for keq in terms of the final concentrations (using zeta).
        '''
               
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
            self.initial_guess = [0]*len(self.keq_values) # All zeros is usually a stationary point- guess strongly reccomended!
        zeta_values = fsolve(self.setup_keq_expressions, self.initial_guess)
        print "The calculated extents of reaction are %s" %(zeta_values)
       
        return zeta_values

    
    def solve_final_concentrations(self):
        '''
        Runs the class.
        '''

        zeta_values = self.get_zeta_values()
        final_concentrations = self.setup_keq_expressions(zeta_values, return_value = 'mol_exps')
            
        print 'The final concentrations are: %s' %(final_concentrations)

