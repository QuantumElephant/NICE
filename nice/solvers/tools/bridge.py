#!/usr/bin/env python

from __future__ import division

import numpy as np


# Although this code can be written more compactly without using a class, structuring things this
# way allows for more functionality to be easily added later on (e.g. guesses for new solvers,
# comparing results from one solver to another, using a partial calculation from one solver and finishing
# the calculation with another solver, etc.)
      
class BridgeSolvers(object):
    '''
    Contains misc. functions to allow for communication (e.g. generate initial guesses) between the 
    different solvers.
    '''

    def __init__(self, initial_concentrations, final_concentrations, stoich_coeff):
        '''
        Arguments:
        ----------
        initial_concentrations: list
            A vector containing the initial concentrations of each species. Order of species in this 
            list should correspond to that of the stoich_coeff columns. In Molar (mol/L).
        final_concentrations: list
             The final concentrations calculated using the Monte-Carlo solver/ exact solver/ other source.
        stoich_coeff: numpy array
            Each column represents a species number, while each row represents a reversible reaction.
            Coefficients are negative if the species is a reactant, positive if the species is a 
            product, and zero if the species doesn't participate in the reaction at all. Every species 
            must be part of each reaction row, even if it doesn't participate in the reaction (the 
            coefficient will be zero in that case.
        '''

        if isinstance(initial_concentrations, list):
            self.initial_concentrations = initial_concentrations
        else:
            raise ValueError, 'The initial concentrations must be in a list format.'

        if isinstance(final_concentrations, list):
            self.final_concentrations = final_concentrations
        else:
            raise ValueError, 'The final concentrations must be in a list format.'
        
        if isinstance(stoich_coeff, np.ndarray):
            self.stoich_coeff = stoich_coeff
        elif isinstance(stoich_coeff, list):
            try:
                stoich_coeff = np.array(stoich_coeff)
            except:
                raise ValueError, 'Please check your coefficient matrix- could not convert to an array.'
        else:
            raise ValueError, 'Please ensure that the coefficient matrix is in the correct format.'


    def get_zeta_guess(self):
        '''
        Uses the intitial and final concentrations to return the reaction extents (zeta values) for each
        reaction. Intended to allow the user to take the results from the Monte-Carlo solver and use them as
        guesses for the exact solver.

        Returns:
        --------
        zeta_guess: numpy array
            The guesses for each zeta value given certain initial and final concentrations. Can be fed into
            the exact equillibria solver directly as an intial guess.
        '''
        coeff_vector = []
        for initial, final in zip(self.initial_concentrations, self.final_concentrations):
            coeff_vector.append(final-initial)

        zeta_guess = np.linalg.lstsq(np.transpose(self.stoich_coeff), coeff_vector)[0]

        return zeta_guess

