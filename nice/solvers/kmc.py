#!/usr/bin/env python

from __future__ import division

import numpy as np
from random import random


class KMCSolver(object):
    '''
    The Kinetic Monte Carlo Solver.
    '''

    def __init__(self, initial_concentrations, keq_values, stoich_coeff, phi=1, concentration_step=1.0e-7):
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

        if stoich_coeff.shape[0] != keq_values.shape[0]:
            raise ValueError('The number of rows in stoich_coeff array should equal the length of keq_values array.')

        if stoich_coeff.shape[1] != initial_concentrations.shape[0]:
            raise ValueError('The number of columns in stoich_coeff array should equal the length of the initial_concentrations array.')

        self.initial_concentrations = initial_concentrations
        self.keq_values = keq_values
        self.stoich_coeff = stoich_coeff
        self.phi = phi
        self.concentration_step = concentration_step

        self.concentrations = initial_concentrations # The initial_concentrations attribute shouldn't be overwritten


    def get_rate_constants(self):
	'''
        Return an estimate of the forward and reverse rate constants.

        Returns:
        --------
        forward_rate_consts: np.ndarray
            The forward rate constants used to calculate forward rates. Order is the same as the keq_values
            order.
        reverse_rate_consts: np.ndarray
            The reverse rate constants used to calculate reverse rates. Order is the same as the keq_values
            order.
	'''
        reverse_rate_consts = self.phi / (self.keq_values + 1.0)
        forward_rate_consts = self.phi - reverse_rate_consts
        self.forward_rate_consts = forward_rate_consts
        self.reverse_rate_consts = reverse_rate_consts
        print 'The forward rate constants are: %s' % (forward_rate_consts)
        print 'The reverse rate constants are: %s' % (reverse_rate_consts)
        return forward_rate_consts, reverse_rate_consts


    def get_rates(self):
        '''
        Uses reactant concetrations and rate constants to determine forward/reverse reaction rates.

        Returns:
        --------
        forward_rates: np.ndarray
            The forward rates for each reaction. Should be updated every iteration.
        reverse_rates: np.ndarray
            The reverse rates for each reaction. Should be updated every iteration.
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

        self.forward_rates = np.array(forward_rates)
        self.reverse_rates = np.array(reverse_rates)

        print 'The forward rates: %s' %(forward_rates)
        print 'The reverse rates: %s' %(reverse_rates)
        return np.array(forward_rates), np.array(reverse_rates)


    def get_net_rates(self):
        '''
        Uses forward and reverse rates to determine the net reaction rate/direction

        Returns:
        --------
        net_rates: np.ndarray
            The forward rate - reverse rate for each reaction. A negative value means that the reverse
            reaction occurs, and a positive value represents a forward reaction.
        '''
        net_rates = self.forward_rates - self.reverse_rates
        print 'Net rates: %s' %(net_rates)
        self.net_rates = net_rates
        return net_rates


    def create_probability_vector(self):
        '''
        Creates a probability vector for plain Monte-Carlo.

        Returns:
        --------
        probaility_vector: list
            Contains cumulative probabilities for the ith forward/reverse reaction to occur.
        '''
        rates = np.append(self.forward_rates, self.reverse_rates)
        probability_vector = np.cumsum(rates)
        # Normalize the probability vector
        probability_vector /= probability_vector[-1]
        self.probability_vector = probability_vector
        return probability_vector


    def create_net_rate_probability_vector(self):
        '''
        Creates a vector of total length 1, with the ith entry correponding to the probability of the ith reaction occuring.

        Returns:
        --------
        probability_vector: list
            Contains cumulative probabilities for the ith reaction to occur.
        '''
        probability_vector = np.cumsum(abs(self.net_rates))
        probability_vector /= probability_vector[-1]
        self.probability_vector = probability_vector
        print 'Prob vector: %s' %(probability_vector)
        print 'Prob vector sum: %s' %(sum(probability_vector))
        return probability_vector


    # Move this into anoter function
    def select_random_value(self):
        '''
        Generates a random number between 0 and 1, used to determine which reaction occurs.

        Returns:
        --------
        r: float
            A number between 0 and 1.
        '''

        r = random()

        self.r = r
        return r


    def select_reaction(self):
        '''
        Determines the reaction that will occur.

        Compares r to each cumulative probability in probability vector to determine which
        reaction will occur in this iteration.

        Returns:
        --------
        selected_rxn: int
            Corresponds to the stoich_coeff row index number of the reaction to be performed.
        '''

        rxn_index = np.where(self.probability_vector > self.r)[0][0]
        self.selected_rxn = rxn_index
        print 'Selected rxn: %s' %(self.selected_rxn)
        print type(self.selected_rxn)
        return rxn_index


    def do_reaction(self, verbosity = 'on'):
        '''
        Changes the concentrations attribute according to which forward/reverse reaction is chosen.
        '''

        nforward_rxns = int(len(self.probability_vector)/2)
        print self.selected_rxn
        print self.selected_rxn - nforward_rxns
        if self.selected_rxn < nforward_rxns: # Checks for a forward rxn
            self.concentrations += self.stoich_coeff[self.selected_rxn] * self.concentration_step

        elif self.selected_rxn >= nforward_rxns: #Checks for a reverse rxn
            self.concentrations -= self.stoich_coeff[self.selected_rxn - nforward_rxns] * self.concentration_step

        if verbosity == 'on':
            print 'The concentrations at the end of this iteration are: %s' %(self.concentrations)


    def do_net_reaction(self, verbosity = 'on'):
        '''
        Changes the concentrations attribute according to which net reaction was selected.

        No returns, but changes the concetrations attribute for each time it is run.
        '''

        for species, coeff in enumerate(self.stoich_coeff[self.selected_rxn]):
            print coeff
            if self.net_rates[self.selected_rxn] >= 0:
                self.concentrations[species] += coeff*self.concentration_step
            elif self.net_rates[self.selected_rxn] < 0:
                self.concentrations[species] -= coeff*self.concentration_step

        if verbosity == 'on':
            print 'The concentrations at the end of this iteration are: %s' %(self.concentrations)

    # MOve as musch as possible to init
    def run_simulation(self, maxiter, net_rate_KMC = False):
        '''
        Runs the calculation.

        Arguments:
        ----------

        maxiter: int
            The number of iterations the algorithm will run for.
        '''

        self.get_rate_constants()
        i = 0
        while i < maxiter:
            self.get_rates()
            if net_rate_KMC == True:
                self.get_net_rates()
                self.create_net_rate_probability_vector()
                self.select_random_value()
                self.select_reaction()
                self.do_net_reaction()
            elif net_rate_KMC == False:
                self.create_probability_vector()
                self.select_random_value()
                self.select_reaction()
                self.do_reaction()
            i +=1
