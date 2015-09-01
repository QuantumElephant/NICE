#!/usr/bin/env python
'''Base class for simultaneous equilibrium solvers.'''


import numpy as np


__all__ = ['BaseSolver']


class BaseSolver(object):
    '''
    Base class for simultaneous equilibrium solvers.
    '''
    def __init__(self, initial_concentrations, keq_values, stoich_coeff):
	'''
	Arguments:
	----------
        initial_concentrations: np.ndarray
            The number of molecules of each reagent. The length must be == to the number of columns in
            the stoich_coeff array. The entry order must be the same as the reagent order in stoich_coeff.
        keq_values: np.ndarray
            The equillibrium constants for each reaction. The length must be == to the number of rows in
            stoic_coeff. The order of keq values must correspond to the reaction order in stoich_coeff.
        stoich_coeff: np.ndarray
            Each column represents a species number, while each row represents a reversible reaction.
            Coefficients are negative if the species is a reactant, positive if the species is a
            product, and zero if the species doesn't participate in the reaction at all. Every species
            must be part of each reaction row, even if it doesn't participate in the reaction (the
            coefficient will be zero in that case.
	'''

        if not (isinstance(initial_concentrations, np.ndarray) and initial_concentrations.ndim == 1):
            raise ValueError('Argument initial_concentrations should be a 1D array.')

        if not (isinstance(keq_values, np.ndarray) and keq_values.ndim == 1):
            raise ValueError('Argument keq_values should be a 1D array.')

        if not (isinstance(stoich_coeff, np.ndarray) and stoich_coeff.ndim == 2):
            raise ValueError('Argument stoich_coeff should be a 2D array.')

        if stoich_coeff.shape[0] != keq_values.shape[0]:
            raise ValueError('The number of rows in stoich_coeff array should equal the length of keq_values array.')

        if stoich_coeff.shape[1] != initial_concentrations.shape[0]:
            raise ValueError('The number of columns in stoich_coeff array should equal the length of the initial_concentrations array.')

        self._initial_concentrations = initial_concentrations
        self._keq_values = keq_values
        self._stoich_coeff = stoich_coeff


    @property
    def initial_concentrations(self):
        '''Return the initial concentration of species.'''
        return self._initial_concentrations

    @property
    def keq_values(self):
        '''Return the equilibrium constant of reactions.'''
        return self._keq_values

    @property
    def stoich_coeff(self):
        '''Return the stoichiometric coefficient of species in every reaction.'''
        return self._stoich_coeff
