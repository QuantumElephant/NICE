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
from scipy.optimize import fsolve

from nice.base import BaseSolver


__all__ = [
    'ExactSolver',
    ]


class ExactSolver(BaseSolver):
    """
    Exact simultaneous equilibrium solver class.

    """

    def run(self, guess, maxiter=1000, tol=1.0e-9):
        """
        Run the exact solver.

        Parameters
        ----------
        guess : np.ndarray(m)
            Guess for the zeta values of each reaction.
        maxiter : int, default=1000
            Maximum number of iterations to perform.
        tol : float, default=1.0e-9
            Convergence tolerance.

        """
        # Handle ``guess`` argument
        guess = np.array(guess, dtype=float)
        if guess.shape != self.keq_values.shape:
            raise ValueError("'guess' must be of the same shape as 'keq_values'")
	# Solve for zeta
        zeta = fsolve(self._keq_expressions, guess, maxfev=maxiter, xtol=tol)
	# Substitute back to get final concentrations
        self._concs = self._mol_expressions(zeta)

    def _mol_expressions(self, zeta):
        """
        Compute the final concentrations of each species from the reaction extents.

        Parameters
        ----------
        zeta : np.ndarray(m)
            Reaction extents for each reaction.

        Returns
        -------
        mol_exps : np.ndarray(m)
            Expressions for the final concentration of each species.

        """
        # Compute conc_n = (init_conc)_n + sum_m { C_mn zeta_m }
        mol_exps = np.empty_like(self._initial_concs)
        for species, coeffs in enumerate(self._stoich_coeffs.T):
            mol_exps[species] = np.sum(coeffs * zeta)
        mol_exps += self._initial_concs
        return mol_exps

    def _keq_expressions(self, zeta):
        """
        Compute the ``keq`` values of each reaction from the reaction extents.

        Parameters
        ----------
        zeta : np.ndarray(m)
            Reaction extents for each reaction.

        Returns
        -------
        keq_exps : np.ndarray(m)
            Expressions for the ``keq`` values in terms of final concentrations.

        """
        # Compute keq_m = prod_n { (conc_n)^(C_mn) }
        mol_exps = self._mol_expressions(zeta)
        keq_exps = np.empty_like(self._keq_values)
        for rxn, coeffs in enumerate(self._stoich_coeffs):
            keq_exps[rxn] = np.prod(np.power(mol_exps, coeffs))
        keq_exps -= self._keq_values
        return keq_exps
