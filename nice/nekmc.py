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
from nice import _nekmc


__all__ = [
    'NEKMCSolver',
    ]


class NEKMCSolver(BaseSolver):
    """
    Net Event Kinetic Monte Carlo simultaneous equilibrium solver class.

    """

    def __init__(self, initial_concs, keq_values, stoich_coeffs, phi=1.0):
        """
        Initialize the Net Event Kinetic Monte Carlo equilibrium solver.

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
        phi : int, default=1.0
            Constant used for constructing the forward and reverse rate
            constants (phi == k_{+1} + k_{-1}).

        """
        # Initialize superclass
        super(NEKMCSolver, self).__init__(initial_concs, keq_values, stoich_coeffs)
        # Compute forward and reverse rate constants
        self._rev_consts = phi / (self._keq_values + 1.0)
        self._fwd_consts = phi - self._rev_consts
        # Initialize reaction rate arrays
        self._fwd_rates = np.zeros_like(self._keq_values)
        self._rev_rates = np.zeros_like(self._keq_values)
        self._net_rates = np.zeros_like(self._keq_values)
        # Compute forward/reverse/net rates
        nreaction, nspecies = self._stoich_coeffs.shape
        _nekmc.update_rates(nspecies, nreaction,
                          self._concs, self._stoich_coeffs.T,
                          self._fwd_consts, self._rev_consts,
                          self._fwd_rates, self._rev_rates, self._net_rates)

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

    def run(self, step=1.0e-6, maxiter=50000):
        """
        Run the NEKMC simulation.

        Parameters
        ----------
        step : float, default=1.0e-6
            Step size for change in concentration at each iteration.
        maxiter : int, default=50000
            Number of iterations to run.

        """
        nreaction, nspecies = self._stoich_coeffs.shape
        _nekmc.run_nekmc(nspecies, nreaction,
                     self._concs, self._stoich_coeffs.T,
                     self._fwd_consts, self._rev_consts,
                     self._fwd_rates, self._rev_rates, self._net_rates,
                     step, maxiter)

    def compute_zeta(self):
        """
        Return the reaction extents (zeta values) for each reaction.

        The result of this method can be fed into the ``ExactSolver`` as an
        initial guess.

        Returns
        -------
        zeta : np.ndarray(m)
            Zeta values for each reaction.

        """
        diff = self._concs - self._initial_concs
        return np.linalg.lstsq(self._stoich_coeffs.T, diff, rcond=None)[0]
