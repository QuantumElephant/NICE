# Copyright (C) 2018 Ayers Lab.
#
# This file is part of NICE.
#
# NICE is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free
# Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# NICE is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
# for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>.


from __future__ import division

import numpy as np

from nice.base import BaseSolver


__all__ = [
    'KMCSolver',
    'NEKMCSolver',
    ]


class KMCSolver(BaseSolver):
    """
    Kinetic Monte Carlo simultaneous equilibrium solver class.

    """

    def __init__(self, initial_concs, keq_values, stoich_coeffs, phi=1, conc_step=1e-7):
        """
        Initialize the Kinetic Monte Carlo equilibrium solver.

        Parameters
        ----------
        initial_concs : np.ndarray((n,))
            Concentration of each species.
            Length ``n`` specifies the number of species.
        keq_values : np.ndarray((m,))
            Equilibrium constant of each reaction. The order of ``keq``
            values must correspond to the reaction order in ``stoich_coeffs``.
            Length ``m`` specifies the number of reversible reactions.
        stoich_coeffs : np.ndarray((m, n))
            Stoichiometric coefficients of each species in each reaction.
            Each row represents a reaction, and each column is a species.
            Coefficients are negative if the species is a reactant and
            positive if it is a product. If the species does not participate
            in a reaction, then that coefficient is zero.
        phi : int, default=1
            Constant value used for constructing a second equation to solve for
            the forward and reverse rate constants (phi == k+1 + k-1). We make
            this approximation because we don't care about reaction kinetics,
        conc_step : float, default=1e-7
            The step size for change in concentration at each iteration.
            Final concentrations become more precise for smaller values of
            ``conc_step``. If it is too small, though, it may fail to converge
            to a solution. Typical values are 6 orders of magnitude below your
            highest initial concentration, and at least 3 to 5 orders less than
            your lowest concentration.

        """
        # Initialize superclass
        super(KMCSolver, self).__init__(initial_concs, keq_values, stoich_coeffs)
        # Set attributes
        self._phi = phi
        self._conc_step = conc_step
        # Compute forward and reverse rate constants
        self._rev_consts = phi / (self._keq_values + 1.)
        self._fwd_consts = phi - self._rev_consts
        # Initialize reaction rate arrays
        nreaction = self._stoich_coeffs.shape[0]
        self._ratearray = np.empty(2 * nreaction)
        self._fwd_rates = self._ratearray[:nreaction]
        self._rev_rates = self._ratearray[nreaction:]
        self._net_rates = np.empty(nreaction)
        # Compute forward/reverse/net rates
        self._update_rates()

    @property
    def fwd_rate_consts(self):
        """
        Return the rate constants for forward reactions.

        """
        return self._fwd_consts

    @property
    def rev_rate_consts(self):
        """
        Return the rate constants for the reverse reactions.

        """
        return self._rev_consts

    @property
    def fwd_rates(self):
        """
        Return the forward rates of reaction.

        """
        return self._fwd_rates

    @property
    def rev_rates(self):
        """
        Return the reverse rates of reaction.

        """
        return self._rev_rates

    @property
    def net_rates(self):
        """
        Return the net rates of reaction.

        """
        return self._net_rates

    def run(self, maxiter=1000):
        """
        Run the KMC simulation.

        Parameters
        ----------
        maxiter : int, default=1000
            Number of iterations to run.

        """
        niter = 0
        while niter < maxiter:
            self._update_rates()
            index = self._select_reaction()
            self._do_reaction(index)
            niter += 1

    def _update_rates(self):
        """
        Compute reaction rates based on rate constants and current concentrations.

        """
        for rxn, coeffs in enumerate(self._stoich_coeffs):
            # Compute forward rate of reaction ``rxn``
            ltzero = coeffs < 0
            self._fwd_rates[rxn] = self._fwd_consts[rxn] \
                * np.prod(np.power(self._concs[ltzero], np.abs(coeffs[ltzero])))
            # Compute reverse of reaction ``rxn``
            gtzero = coeffs > 0
            self._rev_rates[rxn] = self._rev_consts[rxn] \
                * np.prod(np.power(self._concs[gtzero], coeffs[gtzero]))
            # Update net rates of reaction
        self._net_rates = self._fwd_rates - self._rev_rates

    def _select_reaction(self):
        """
        Select a forward or reverse reaction to occur.

        Compares rate to each cumulative rate probability to determine which
        reaction will occur.

        Returns
        -------
        index : int
            Index of reaction to occur.

        """
        # Generate index via probability vector from [forward, reverse] rates
        # Forward reaction indices come before reverse reaction indices
        pvec = np.cumsum(self._ratearray)
        pvec /= pvec[-1]
        return np.where(pvec > np.random.random())[0][0]

    def _do_reaction(self, index):
        """
        Change the concentrations according to a forward or reverse reaction.

        Parameters
        ----------
        index : int
            Index of reaction to occur.

        """
        # Change the concentrations of species involved in reaction
        # Forward reaction indices come before reverse reaction indices
        nforward = self._fwd_rates.size
        if index < nforward:
            self._concs += self._stoich_coeffs[index] * self._conc_step
        else:
            self._concs -= self._stoich_coeffs[index - nforward] * self._conc_step


class NEKMCSolver(KMCSolver):
    """
    Net Event Kinetic Monte Carlo simultaneous equilibrium solver class.

    """

    def _select_reaction(self):
        """
        Select a net reaction to occur.

        Compares rate to each cumulative rate probability to determine which
        reaction will occur.

        Returns
        -------
        index : int
            Index of reaction to occur.

        """
        # Generate index via probability vector from rates
        pvec = np.cumsum(np.abs(self._net_rates))
        pvec /= pvec[-1]
        return np.where(pvec > np.random.random())[0][0]

    def _do_reaction(self, index):
        """
        Change the concentrations according to a net reaction.

        Parameters
        ----------
        index : int
            Index of reaction to occur.

        """
        # Change the concentrations of species involved in reaction
        for species, coeff in enumerate(self._stoich_coeffs[index]):
            if self._net_rates[index] >= 0:
                self._concs[species] += coeff * self._conc_step
            else:
                self._concs[species] -= coeff * self._conc_step
