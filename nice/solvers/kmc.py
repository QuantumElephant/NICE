#!/usr/bin/env python
'''Kinetic Monte Carlo simultaneous equilibrium solver.'''

from __future__ import division
import numpy as np
from random import random
from nice.solvers.base import BaseSolver


class KMCSolver(BaseSolver):
    '''
    The Kinetic Monte Carlo (KMC) simultaneous equilibrium solver.
    '''

    def __init__(self, initial_concentrations, keq_values, stoich_coeff, phi=1, concentration_step=1.0e-7, net_rxn=False):
        '''
        Arguments:
        ----------
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
        super(KMCSolver, self).__init__(initial_concentrations, keq_values, stoich_coeff)

        if not isinstance(phi, int):
            raise ValueError('Argument phi should be an integer.')

        if not isinstance(concentration_step, float):
            raise ValueError('Argument concentration_step should be a float.')

        if not isinstance(net_rxn, bool):
            raise ValueError('Argument net_rxn should be a boolean.')

        self._phi = phi
        self._concentration_step = concentration_step
        self._net_rxn = net_rxn

        # Calculate the forward and reverse rate constants (k)
        self._reverse_rate_consts = self._phi / (self._keq_values + 1.0)
        self._forward_rate_consts = self._phi - self._reverse_rate_consts

        # Set the concentrations and calculate the forward/reverse/net rates
        self._concentrations = np.copy(self._initial_concentrations)
        self._forward_rates, self._reverse_rates, self._net_rates = self.calculate_rates()

    @property
    def phi(self):
        '''Return the phi.'''
        return self._phi

    @property
    def concentration_step(self):
        '''Return the amount of change in concentration.'''
        return self._concentration_step

    @property
    def net_rxn(self):
        ''' '''
        return self._net_rxn

    @property
    def forward_rate_consts(self):
        '''Return the rate constant for the forward reactions.'''
        return self._forward_rate_consts

    @property
    def reverse_rate_consts(self):
        '''Return the rate constant for the reverse reactions.'''
        return self._reverse_rate_consts

    @property
    def concentrations(self):
        '''Return the current concentration of species.'''
        return self._concentrations

    @property
    def forward_rates(self):
        '''Return the forward rate of reactions.'''
        return self._forward_rates

    @property
    def reverse_rates(self):
        '''Return the reverse rate of reactions.'''
        return self._reverse_rates

    @property
    def net_rates(self):
        '''Return the net rate of reactions.'''
        return self._net_rates


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
        for i, coeffs in enumerate(self._stoich_coeff):
            # calculating the forward rate of reaction i
            forward_rate = np.array([np.power(self._concentrations[j], abs(coeff)) for j, coeff in enumerate(coeffs) if coeff < 0])
            forward_rate = self._forward_rate_consts[i] * np.prod(forward_rate)
            forward_rates.append(forward_rate)
            # calculating the reverse rate of reactions i
            reverse_rate = np.array([np.power(self._concentrations[j], coeff) for j, coeff in enumerate(coeffs) if coeff > 0])
            reverse_rate = self._reverse_rate_consts[i] * np.prod(reverse_rate)
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
        if self._net_rxn:
            rates = abs(self._net_rates)
        else:
            rates = np.append(self._forward_rates, self._reverse_rates)
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
        if self._net_rxn:
            # net reaction indexed rxn_index changes the concentrations
            for species, coeff in enumerate(self._stoich_coeff[rxn_index]):
                if self._net_rates[rxn_index] >= 0:
                    self._concentrations[species] += coeff*self._concentration_step
                elif self._net_rates[rxn_index] < 0:
                    self._concentrations[species] -= coeff*self._concentration_step
        else:
            nforward_rxns = int(len(self._forward_rates))
            if rxn_index < nforward_rxns:
                # forward reaction indexed rxn_index changes the concentrations
                self._concentrations += self._stoich_coeff[rxn_index] * self._concentration_step

            elif rxn_index >= nforward_rxns:
                # reverse reaction indexed rxn_index changes the concentrations
                self._concentrations -= self._stoich_coeff[rxn_index - nforward_rxns] * self._concentration_step


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
            self._forwrd_rates, self._reverse_rates, self._net_rates = self.calculate_rates()
            # select the reaction to happen
            selected_index = self.select_reaction()
            # have the selected reaction change concentrations
            self.do_reaction(selected_index)
            print 'iterations:', i
            print 'concentration:', self._concentrations
            print 'forward rates:', self._forward_rates
            print 'reverse rates:', self._reverse_rates
            print 'net     rates:', self._net_rates
            print
            i +=1
