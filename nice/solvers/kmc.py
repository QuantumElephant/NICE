#!/usr/bin/env python

from __future__ import division

import numpy as np
from random import random


class KMCSolver(object):
    '''
    The Kinetic Monte Carlo Solver.
    '''

    def __init__(self, initial_concentrations, keq_values, stoich_coeff, phi=1, concentration_step=1.0e-7, net_rxn=False):
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
        phi: int
            Used as a constant value in order to construct a second equation to solve for the forward
            and reverse rate constants. phi == k1 + k-1. Able to make this approximation since we don't
            care about reaction kinetics, only final values.
        concentration_step: float
            The value that concentrations can change during every iteration. As this value is decreased,
            final concentration values are more precise. However, if this number is too small, the solver
            will take a long time to converge to a solution. Typically 10e-6 orders of magnitude lower
            than your highest concentration, and at least 10e-3 to 10e-5 less than your lowest concentration.
	'''

        if not (isinstance(initial_concentrations, np.ndarray) and initial_concentrations.ndim == 1):
            raise ValueError('Argument initial_concentrations should be a 1D array.')

        if not (isinstance(keq_values, np.ndarray) and keq_values.ndim == 1):
            raise ValueError('Argument keq_values should be a 1D array.')

        if not (isinstance(stoich_coeff, np.ndarray) and stoich_coeff.ndim == 2):
            raise ValueError('Argument stoich_coeff should be a 2D array.')

        if not isinstance(phi, int):
            raise ValueError('Argument phi should be an integer.')

        if not isinstance(concentration_step, float):
            raise ValueError('Argument concentration_step should be a float.')

        if not isinstance(net_rxn, bool):
            raise ValueError('Argument net_rxn should be a boolean.')

        if stoich_coeff.shape[0] != keq_values.shape[0]:
            raise ValueError('The number of rows in stoich_coeff array should equal the length of keq_values array.')

        if stoich_coeff.shape[1] != initial_concentrations.shape[0]:
            raise ValueError('The number of columns in stoich_coeff array should equal the length of the initial_concentrations array.')

        self.initial_concentrations = initial_concentrations
        self.keq_values = keq_values
        self.stoich_coeff = stoich_coeff
        self.phi = phi
        self.concentration_step = concentration_step
        self.net_rxn = net_rxn

        # Calculate the forward and reverse rate constants (k)
        self.reverse_rate_consts = self.phi / (self.keq_values + 1.0)
        self.forward_rate_consts = self.phi - self.reverse_rate_consts

        # Set the concentrations and calculate the forward/reverse/net rates
        self.concentrations = np.copy(self.initial_concentrations)
        self.forward_rates, self.reverse_rates, self.net_rates = self.calculate_rates()


    def calculate_rates(self):
        '''
        Return the forward/reverse/net reaction rates baed on the rate constants and current concentrations.

        Returns:
        --------
        forward_rates: np.ndarray
            The forward rates for each reaction. Should be updated every iteration.
        reverse_rates: np.ndarray
            The reverse rates for each reaction. Should be updated every iteration.
        net_rates: np.ndarray
            The forward rate - reverse rate for each reaction. A negative value means that the reverse
            reaction occurs, and a positive value represents a forward reaction.
        '''
        forward_rates = []
        reverse_rates = []
        for i, coeffs in enumerate(self.stoich_coeff):
            # calculating the forward rate of reaction i
            forward_rate = np.array([np.power(self.concentrations[j], abs(coeff)) for j, coeff in enumerate(coeffs) if coeff < 0])
            forward_rate = self.forward_rate_consts[i] * np.prod(forward_rate)
            forward_rates.append(forward_rate)
            # calculating the reverse rate of reactions i
            reverse_rate = np.array([np.power(self.concentrations[j], coeff) for j, coeff in enumerate(coeffs) if coeff > 0])
            reverse_rate = self.reverse_rate_consts[i] * np.prod(reverse_rate)
            reverse_rates.append(reverse_rate)
        forward_rates = np.array(forward_rates)
        reverse_rates = np.array(reverse_rates)
        net_rates = forward_rates - reverse_rates
        return forward_rates, reverse_rates, net_rates


    def create_probability_vector(self):
        '''
        Creates a probability vector for plain Monte-Carlo.

        Returns:
        --------
        probaility_vector: np.ndarray
            Contains cumulative probabilities for the ith forward/reverse reaction to occur.
        '''
        if self.net_rxn:
            rates = abs(self.net_rates)
        else:
            rates = np.append(self.forward_rates, self.reverse_rates)
        # Cumulative sum of rates
        probability_vector = np.cumsum(rates)
        # Normalize the probability vector
        probability_vector /= probability_vector[-1]
        return probability_vector


    def select_reaction(self, rate=None):
        '''
        Return the index of the reaction that will occur.

        Compares rate to each cumulative rate probability to determine which
        reaction will occur.

        Returns:
        --------
        selected_rxn: int
            Corresponds to the stoich_coeff row index number of the reaction to be performed.
        '''
        # Calculate the rate probability
        probability_vector = self.create_probability_vector()
        if rate is None:
            rate = random()
        else:
            assert isinstance(rate, float)
        # Select the reaction
        rxn_index = np.where(probability_vector > rate)[0][0]
        return rxn_index


    def do_reaction(self, rxn_index):
        '''
        Changes the concentrations according to which forward/reverse/net reaction is chosen.

        No returns, but changes the concetrations attribute for each time it is run.
        '''
        if self.net_rxn:
            # net reaction indexed rxn_index changes the concentrations
            for species, coeff in enumerate(self.stoich_coeff[rxn_index]):
                if self.net_rates[rxn_index] >= 0:
                    self.concentrations[species] += coeff*self.concentration_step
                elif self.net_rates[rxn_index] < 0:
                    self.concentrations[species] -= coeff*self.concentration_step
        else:
            nforward_rxns = int(len(self.forward_rates))
            if rxn_index < nforward_rxns:
                # forward reaction indexed rxn_index changes the concentrations
                self.concentrations += self.stoich_coeff[rxn_index] * self.concentration_step

            elif rxn_index >= nforward_rxns:
                # reverse reaction indexed rxn_index changes the concentrations
                self.concentrations -= self.stoich_coeff[rxn_index - nforward_rxns] * self.concentration_step


    def run_simulation(self, maxiter):
        '''
        Runs the KMC simulation in every step of which the concentraions and rates change.

        Arguments:
        ----------

        maxiter: int
            The number of iterations the algorithm will run for.
        '''
        # The concentrations and forward/reverse/net rates change as the simulation runs
        i = 0
        while i < maxiter:
            # update rates
            self.forwrd_rates, self.reverse_rates, self.net_rates = self.calculate_rates()
            # select the reaction to happen
            selected_index = self.select_reaction()
            # have the selected reaction change concentrations
            self.do_reaction(selected_index)
            print 'iterations:', i
            print 'concentration:', self.concentrations
            print 'forward rates:', self.forward_rates
            print 'reverse rates:', self.reverse_rates
            print 'net     rates:', self.net_rates
            print
            i +=1
