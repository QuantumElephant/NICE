#!/usr/bin/env python
'''Exact simultaneous equilibrium solver.'''

from __future__ import division

import numpy as np
from scipy.optimize import fsolve

from nice.solvers.base import BaseSolver


class ExactEqmSolver(BaseSolver):
    '''Solves for final concentrations by solving a system of nonlinear equations'''

    def __init__(self, initial_concentrations, keq_values, stoich_coeff, initial_guess=None):
        '''
        Arguments:
        ----------
        initial_guess: list
            Guess for the zeta values of each reaction, but *not* the initial concentrations.
        '''
        super(ExactEqmSolver, self).__init__(initial_concentrations, keq_values, stoich_coeff)

        if initial_guess is None:
            # Using zeros for initial guess is usually a stationary point.
            # Using a proper guess is strongly recommended.
            initial_guess = np.zeros(self._keq_values.shape)

        if not (isinstance(initial_guess, np.ndarray) and initial_guess.ndim == 1):
            raise ValueError('Argument initial_guess should be a 1D array.')

        if initial_guess.shape != self._keq_values.shape:
            raise ValueError('The shape of initial_guess array should equal that of keq_values array.')

        self._initial_guess = initial_guess

    @property
    def initial_guess(self):
        '''Return the initial guess.'''
        return self._initial_guess

    def setup_keq_expressions(self, z, **kwargs):
        '''
        Creates the correct keq expressions in terms of the final concentrations for each species.

        Arguments:
        ----------
        z: list
            Each element represents a different zeta value.
        kwargs: dict
            Used to alter the return value of the function (to get the mol_exps back instead of keq_exps).
            The only acceptable keyword currently is 'return_value' and the corresponding value is 'mol_exps'.
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
        for species, concentration in enumerate(self._initial_concentrations):
            mol_exp = concentration
            for i, coeff in enumerate(np.nditer(self._stoich_coeff[:,species])):
                mol_exp = mol_exp + coeff*z[i]
            mol_exps.append(mol_exp)


        keq_exps = []
        for rxn, keq in enumerate(self._keq_values):
            keq_exp = 1
            for expr, coeff in zip(mol_exps, np.nditer(self._stoich_coeff[rxn,:])):
                term = expr**coeff
                keq_exp = keq_exp*term
            keq_exp = keq_exp - self._keq_values[rxn]
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

        zeta_values = fsolve(self.setup_keq_expressions, self._initial_guess)
        print "The calculated extents of reaction are %s" %(zeta_values)

        return zeta_values


    def solve_final_concentrations(self):
        '''
        Runs the class.
        '''
        zeta_values = self.get_zeta_values()
        final_concentrations = self.setup_keq_expressions(zeta_values, return_value = 'mol_exps')

        print 'The final concentrations are: %s' %(final_concentrations)

        return final_concentrations
