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

from scipy.optimize import fsolve, fmin_slsqp

from nice.base import BaseSolver


__all__ = [
    'ExactSolver',
    ]


class ExactSolver(BaseSolver):
    r"""
    Exact simultaneous equilibrium solver class.

    Solve the system of equations

    .. math::

        K^m = \prod^N_n { {c_n}^{v^{(m)}_n} }

    where

    .. math::

        c_n = c^{(0)}_n + \sum^M_m { v^{(m)}_n \zeta^{(m)}_n }

    for reaction extents :math:`\zeta^{(m)}_n` and equilibrium concentrations :math:`c_n`.

    """

    def optimize(self, guess: np.ndarray, mode: str = 'newton', maxiter: int = 1000,
        tol: float = 1.0e-9, eps: float = 1.0e-8, sigma: float = 0.5) -> None:
        r"""
        Optimize the solution for the exact solver.

        Parameters
        ----------
        guess : np.ndarray(m)
            Guess for the zeta values of each reaction.
        mode : ('newton' | 'bound' | 'cma'), default='newton'
            Whether to use local Newton optimizer, bounded (concs >= 0) SLSQP optimizer,
            or stochastic CMA optimizer.
        maxiter : int, default=1000
            Maximum number of iterations to perform.
        tol : float, default=1.0e-9
            Convergence tolerance.
        eps : float, default=1.0e-8
            Step size for finite difference derivative approximation. Only used with
            ``mode='newton'``.
        sigma : float, default=0.5
            Initial standard deviation in each coordinate. Only used with ``mode='cma'``.

        """
        # Handle ``guess`` argument
        guess = np.asarray(guess, dtype=np.double)
        if guess.shape != self.keq_values.shape:
            raise ValueError("'guess' must be of the same shape as 'keq_values'")

        # Run optimizer
        if mode == 'newton':
            zeta = fsolve(self._keq_expressions, guess, maxfev=maxiter, xtol=tol, epsfcn=eps)
        elif mode == 'bound':
            f = lambda z: np.sqrt(np.sum(self._keq_expressions(z) ** 2))
            options = {
                'disp': 0,
                'f_ieqcons': self._mol_expressions,
                'iter': maxiter,
                'acc': tol,
                'epsilon': eps,
                }
            zeta = fmin_slsqp(f, guess, **options)
        elif mode == 'cma':
            # Uncommon package; save import for here
            import cma
            f = lambda z: np.sqrt(np.sum(self._keq_expressions(z) ** 2))
            options = {
                'ftarget': 0.0,
                'maxfevals': maxiter,
                'tolx': tol,
                'verbose': -9,
                }
            zeta = cma.fmin2(f, guess, sigma, options=options)[0]
        else:
            raise ValueError("'mode' must be either 'newton', 'bound', or 'cma'")

        # Substitute back to get final concentrations
        self._concs = self._mol_expressions(zeta)

    def _mol_expressions(self, zeta: np.ndarray) -> np.ndarray:
        r"""
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
        for species, coeffs in enumerate(self._stoich_coeffs.transpose()):
            mol_exps[species] = np.sum(coeffs * zeta)
        mol_exps += self._initial_concs
        return mol_exps

    def _keq_expressions(self, zeta: np.ndarray) -> np.ndarray:
        r"""
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
