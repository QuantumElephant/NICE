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


__all__ = [
    'BaseSolver',
    ]


class BaseSolver(object):
    """
    Base simultaneous equilibrium solver class.

    """

    def __init__(self, initial_concs, keq_values, stoich_coeffs):
        """
        Initialize the equilibrium solver.

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

        """
        # Cast to float arrays
        initial_concs = np.array(initial_concs, dtype=float)
        keq_values = np.array(keq_values, dtype=float)
        stoich_coeffs = np.array(stoich_coeffs, dtype=float)
        # Input checking
        if initial_concs.ndim != 1:
            raise ValueError("'initial_concs' must be a 1D array")
        if keq_values.ndim != 1:
            raise ValueError("'keq_values' must be a 1D array")
        if stoich_coeffs.ndim != 2:
            raise ValueError("'stoich_coeffs' must be a 2D array")
        if stoich_coeffs.shape != (keq_values.size, initial_concs.size):
            raise ValueError("'stoich_coeffs' shape must match (nreactions, nreagents)")
        # Set attributes
        self._initial_concs = initial_concs
        self._keq_values = keq_values
        self._stoich_coeffs = stoich_coeffs
        self._concs = np.copy(initial_concs)

    @property
    def initial_concs(self):
        """
        Return the initial concentration of species.

        """
        return self._initial_concs

    @property
    def keq_values(self):
        """
        Return the equilibrium constant of reactions.

        """
        return self._keq_values

    @property
    def stoich_coeffs(self):
        """
        Return the stoichiometric coefficient of species in reactions.

        """
        return self._stoich_coeffs

    @property
    def concs(self):
        """
        Return the concentrations of species.

        """
        return self._concs

    def run(self):
        """
        Run the simulation.

        """
        raise NotImplementedError

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
