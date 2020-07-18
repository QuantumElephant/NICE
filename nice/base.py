# Copyright (C) 2020 Ayers Lab.
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
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
# for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>.

import numpy as np


__all__ = [
    'BaseSolver',
    ]


class BaseSolver:
    r"""
    Simultaneous equilibrium solver class.

    """

    @property
    def nreaction(self) -> int:
        r"""
        Number of reactions for this system.

        """
        return self._stoich_coeffs.shape[0]

    @property
    def nspecies(self) -> int:
        r"""
        Number of species for this system.

        """
        return self._stoich_coeffs.shape[1]

    @property
    def initial_concs(self) -> np.ndarray:
        r"""
        Initial concentration of species.

        """
        return self._initial_concs

    @property
    def stoich_coeffs(self) -> np.ndarray:
        r"""
        Stoichiometric coefficient of species in reactions.

        """
        return self._stoich_coeffs

    @property
    def concs(self) -> np.ndarray:
        r"""
        Concentrations of species.

        """
        return self._concs

    @property
    def keq_values(self) -> np.ndarray:
        r"""
        Equilibrium constant of reactions.

        """
        if self._keq_values is None:
            raise AttributeError("'keq_values' was not provided")
        return self._keq_values

    @property
    def fwd_rate_consts(self) -> np.ndarray:
        r"""
        Rate constants for forward reactions.

        """
        return self._fwd_consts

    @property
    def rev_rate_consts(self) -> np.ndarray:
        r"""
        Rate constants for the reverse reactions.

        """
        return self._rev_consts

    def __init__(self, initial_concs: np.ndarray, stoich_coeffs: np.ndarray,
        keq_values: np.ndarray = None, rate_consts: np.ndarray = None, phi: float = 1.0) -> None:
        r"""
        Initialize the equilibrium solver.

        Parameters
        ----------
        initial_concs : np.ndarray((n,))
            Concentration of each species. Length ``n`` specifies the number of species.
        stoich_coeffs : np.ndarray((m, n))
            Stoichiometric coefficients of each species in each reaction. Each row represents a
            reaction, and each column is a species. Coefficients are negative if the species is a
            reactant and positive if it is a product. If the species does not participate in a
            reaction, then that coefficient is zero.
        keq_values : np.ndarray((m,)), optional
            Equilibrium constant of each reaction. The order of ``keq`` values must correspond to
            the reaction order in ``stoich_coeffs``. Length ``m`` specifies the number of reversible
            reactions.
        rate_consts : np.ndarray((2, m)), optional
            Rate constants for each reaction. The first row specifies the forward rate constants and
            the second row specifies the reverse rate constants.
        phi : int, default=1.0
            Constant used for constructing the forward and reverse rate constants
            (:math:`\phi = k_{+1} + k_{-1}`).

        Notes
        -----
        One of (``keq_values``, ``rate_consts``) _must_ be specified.

        """
        # Check input mode (either ``keq_values`` or ``rate_consts`` specified)
        if keq_values is None:
            if rate_consts is None:
                raise ValueError("One of ('keq_values', 'rate_consts') must be passed as input")
            rate_consts = np.array(rate_consts, dtype=np.double)
            if rate_consts.ndim != 2 or rate_consts.shape[0] != 2:
                raise ValueError("'rate_consts' must be a 2D array with two rows")
            nreaction = rate_consts.shape[1]
        else:
            if rate_consts is not None:
                raise ValueError("Cannot set from both ('keq_values', 'rate_consts')")
            keq_values = np.array(keq_values, dtype=np.double)
            if keq_values.ndim != 1:
                raise ValueError("'keq_values' must be a 1D array")
            nreaction = keq_values.shape[0]

        # Check system arrays
        initial_concs = np.array(initial_concs, dtype=np.double)
        stoich_coeffs = np.array(stoich_coeffs, dtype=np.double)
        if initial_concs.ndim != 1:
            raise ValueError("'initial_concs' must be a 1D array")
        if stoich_coeffs.ndim != 2:
            raise ValueError("'stoich_coeffs' must be a 2D array")
        if stoich_coeffs.shape != (nreaction, initial_concs.size):
            raise ValueError("'stoich_coeffs' shape must match (nreaction, nspecies)")

        # Set system attributes
        self._initial_concs = initial_concs
        self._stoich_coeffs = stoich_coeffs
        self._concs = np.copy(initial_concs)
        self._keq_values = keq_values

        # Set forward and reverse rate constants
        if keq_values is None:
            self._fwd_consts = rate_consts[0]
            self._rev_consts = rate_consts[1]
        else:
            self._rev_consts = phi / (self._keq_values + 1)
            self._fwd_consts = phi - self._rev_consts

    def compute_zeta(self) -> np.ndarray:
        r"""
        Return the reaction extents (zeta values) for each reaction.

        The result of this method can be fed into ``ExactSolver.optimize`` as an initial guess.

        Returns
        -------
        zeta : np.ndarray(m)
            Zeta values for each reaction.

        """
        return np.linalg.lstsq(
            self._stoich_coeffs.transpose(),
            self._concs - self._initial_concs,
            rcond=-1,
            )[0]
