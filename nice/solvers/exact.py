#!/usr/bin/env python

from __future__ import division

import numpy as np
import sympy
from scipy.optimize import fsolve

class ExactEqmSolver(object):
    ''' Solves for final mol fractions, or concentrations through non-linear equations '''

    def __init__(self, initial_concentrations, keq_values, stoich_coeff, initial_guess = None, keq_mol_frac = False):
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
        
        # Checks for/ converts to the right input types.        

        if isinstance(initial_concentrations, list):
            self.initial_concentrations = initial_concentrations
        else:
            raise ValueError, 'Please ensure the intial concentrations are a list.'

        if isinstance(keq_values, list):
            self.keq_values = keq_values
        else:
            raise ValueError, 'Please ensure that your equillibrium constants are a list.'
        
        if isinstance(stoich_coeff, np.ndarray):
            self.stoich_coeff = stoich_coeff
        elif isinstance(stoich_coeff, list):
            try:
                self.stoich_coeff = np.array(stoich_coeff)
            except:
                raise ValueError, 'Please check your coefficient matrix- could not convert to an array.'
        else:
            raise ValueError, 'Please ensure that the coefficient matrix is in the correct format.'

        if isinstance(initial_guess, list):
            self.initial_guess = initial_guess
        else:
            raise ValueError, 'Please ensure that your initial guess values are in a list.'
        
        if isinstance(keq_mol_frac, bool):
            self.keq_mol_frac = keq_mol_frac
        else:
            raise ValueError, 'The format of the equillibrium constants (keq_mol_frac) must be a boolean value.'

        # Check that the dimensions of the coefficient array match with the number of keq values/ species.

        if self.stoich_coeff.shape[0] != len(self.keq_values):
            raise ValueError, 'The number of rows in the coefficient matrix must equal the number of Keq values given.'
        elif self.stoich_coeff.shape[1] != len(self.initial_concentrations):
            raise ValueError, 'The number of columns in the coefficient matrix must equal the number of intial concentrations given.'
 

        self.nspecies = len(self.initial_concentrations)
        self.nreactions = len(self.keq_values)


    def setup_mol_molar_expressions(self):
        '''
        Creates expressions for the final number of moles or molars for each species.

        Note that if the equillibrium constant is in concentrations, the zeta values (extents
        of reaction) are in units of molar. If the equillibrium constant is in terms of mol fractions,
        the zeta values are in units of mol.

        Returns:
        --------
        mol_exps: list
            Contains an equation expressing the final number of moles/ molars for each species using initial
            concentrations, and the number of moles/ molars created/ consumed in each reaction. Equations are
            created using sympy.
        '''
       
        mol_exps = []
        zeta_vars = sympy.symbols('z0:' + str(self.nreactions))
        for x in range(self.nspecies):
            initial = self.initial_concentrations
            x_coeff = self.stoich_coeff[:,x]
            mol_expression = initial[x]
            for i, element in enumerate(np.nditer(x_coeff, order = 'K')):
                mol_expression = mol_expression + zeta_vars[i]*element
            mol_exps.append(mol_expression)
        
        self.zeta_vars = zeta_vars
        self.mol_exps = mol_exps
        return mol_exps

    def get_molfractions(self):
        '''
        Creates mol fraction expressions using the expressions for final moles in terms of zeta.

        Returns:
        --------
        mol_fractions: list
            Expresses the concentration of each species in terms of mol fractions. 
            Expressions are created/ manipulated using sympy symbols.
        '''

        mol_fractions = []
        for x in range(self.nspecies):
            mol_fraction = self.mol_exps[x]/sum(self.mol_exps)
            mol_fractions.append(mol_fraction)

        self.mol_fractions = mol_fractions
        return mol_fractions

    def setup_keq_exps_molfrac(self):
        '''
        Uses the mol fraction expressions to create keq expressions for each reaction.

        If keq is expressed in mol fractions, the mol fraction equations will be used to construct
        the expressions for keq. If keq is expressed in concentrations, then the expressions for 
        each species in terms of concentration will be used.

        Returns:
        --------
        keq_exps: list
            Equillibrium constant expressions for each reaction, expressed in terms of mol fractions.
        '''
        
        keq_exps = []
        for x in range(self.nreactions): # Change to make this loop more pythonic
            x_coeff = self.stoich_coeff[x,:]
            keq_nocoeff = 1
            for (i,v) in zip(self.mol_fractions, np.nditer(x_coeff, order = 'K')):
                term = i**v
                keq_nocoeff = keq_nocoeff*term
            keq_coeff = keq_nocoeff - self.keq_values[x]
            keq_exps.append(keq_coeff)
                
        print keq_exps    
        self.keq_exps = keq_exps
        return keq_exps

    
    def setup_keq_exps_conc(self): 
        '''
        Uses the concentration expressions to create keq expressions for each reaction.

        If keq is expressed in mol fractions, the mol fraction equations will be used to construct
        the expressions for keq. If keq is expressed in concentrations, then the expressions for 
        each species in terms of concentration will be used.

        Returns:
        --------
        keq_exps: list
            Equillibrium constant expressions for each reaction, expressed in terms of concentrations.
        '''
    
        keq_exps = []
        for x in range(self.nreactions): # Change to make this loop more pythonic
            x_coeff = self.stoich_coeff[x,:]
            keq_nocoeff = 1
            for (i,v) in zip(self.mol_exps, np.nditer(x_coeff, order = 'K')):
                term = i**v
                keq_nocoeff = keq_nocoeff*term
            keq_coeff = keq_nocoeff - self.keq_values[x]
            keq_exps.append(keq_coeff)

        self.keq_exps = keq_exps
        return keq_exps

    def convert_scipy_func(self, z):
        '''
        Returns the expressions created using sympy into the correct input format for scipy.optimize solvers.

        Arguments:
        ----------
        z: list
            Each entry represents a different variable z0, z1, z2, etc. Needed to use scipy fsolve (fsolve passes
            values for z).

        Returns:
        --------
        functions: list
            Expressions converted from sympy expressions into a form that scipy will be able to use.
        '''

        functions_sympy = map(str, self.keq_exps)
        functions = []
        for i, func in enumerate(functions_sympy):
            for x in range(len(self.zeta_vars)):
                functions_sympy[i] = functions_sympy[i].replace('z' + str(x), 'z' + '[' + str(x) + ']')
            functions.append(eval(functions_sympy[i])) # Try to think of a better way to do this than using eval!
        
        return functions


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
        zeta_values = fsolve(self.convert_scipy_func, self.initial_guess)
        print "The calculated extents of reaction are %s" %(zeta_values)
       
        self.zeta_values = zeta_values
        return zeta_values


    def get_final_concentrations(self):
        '''
        Uses calculated zeta values into the final concentrations for each species. 

        Returns:
        --------
        final_concentrations: list 
            The final concentration for each species in the reaction network.
        '''

        zeta_vars_values = {}
        for var, value in zip(self.zeta_vars, self.zeta_values):
            zeta_vars_values[var] = value 
        
        final_concentrations = []    
        for equation in self.mol_exps:
            final_concentration = equation.evalf(subs = zeta_vars_values)
            final_concentrations.append(final_concentration)
       
        self.final_concentrations = final_concentrations
        return final_concentrations    
    

    def solve_final_concentrations(self):
        '''
        Runs the class using a single method.
        '''
        
        self.setup_mol_molar_expressions()
        self.get_molfractions()
        if self.keq_mol_frac == True:
            self.setup_keq_exps_molfrac()
        elif self.keq_mol_frac == False:
            self.setup_keq_exps_conc()
        self.get_zeta_values()
        self.get_final_concentrations()

        print 'The final concentrations are: %s' %(self.final_concentrations)

