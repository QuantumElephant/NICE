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

from typing import Tuple

import numpy as np

from nice.base import BaseSolver
from nice._kmc import run_kmc, run_nekmc, update_rates


__all__ = [
    'KMCSolver',
    'NEKMCSolver',
    ]


class BaseKMCSolver(BaseSolver):
    r"""
    Base Kinetic Monte Carlo solver class.

    """

    @property
    def fwd_rates(self) -> np.ndarray:
        r"""
        Forward rates of reaction.

        Returns
        -------
        fwd_rates : np.ndarray((m,))
            Forward rates of reaction.

        """
        return self._fwd_rates

    @property
    def rev_rates(self) -> np.ndarray:
        r"""
        Reverse rates of reaction.

        Returns
        -------
        rev_rates : np.ndarray((m,))
            Reverse rates of reaction.

        """
        return self._rev_rates

    @property
    def net_rates(self) -> np.ndarray:
        r"""
        Net rates of reaction.

        Returns
        -------
        net_rates : np.ndarray((m,))
            Net rates of reaction.

        """
        return self._net_rates

    @property
    def time(self) -> float:
        r"""
        Current time elapsed over past iterations.

        Returns
        -------
        time : float
            Current time elapsed over past iterations.

        """
        return self._time

    def __init__(self, initial_concs: np.ndarray, stoich_coeffs: np.ndarray,
        keq_values: np.ndarray = None, rate_consts: np.ndarray = None, phi: float = 1.0) -> None:
        r"""
        Initialize the KMC system.

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
        # Initialize base class
        BaseSolver.__init__(self, initial_concs, stoich_coeffs, keq_values=keq_values,
            rate_consts=rate_consts, phi=phi)

        # Initialize reaction rate arrays
        self._fwd_rates = np.zeros_like(self._fwd_consts)
        self._rev_rates = np.zeros_like(self._fwd_consts)
        self._net_rates = np.zeros_like(self._fwd_consts)

        # Compute forward/reverse/net rates
        update_rates(
            self.nspecies, self.nreaction,
            self._concs, self._stoich_coeffs.transpose(),
            self._fwd_consts, self._rev_consts,
            self._fwd_rates, self._rev_rates, self._net_rates,
            )

        # Set other persistent KMC attributes
        self._time = 0.0

    def iterate(self, step: float = 1.0e-6, niter: int = 1000) -> None:
        r"""
        Run some iterations of the KMC simulation.

        Parameters
        ----------
        step: float, default=1.0e-6
            Step size for change in concentration at each iteration.
        niter: int, default=1000
            Number of iterations to run.

        """
        self._time += self._iterate(
            self.nspecies, self.nreaction,
            self._concs, self._stoich_coeffs.transpose(),
            self._fwd_consts, self._rev_consts,
            self._fwd_rates, self._rev_rates, self._net_rates,
            step, niter,
            )[3]

    def run_simulation(self, mode: str = 'static', step: float = 1.0e-6, niter: int = 1000,
            maxcall: int = 1000, tol_t: float = 1.0e12, tol_s: float = 1.0e-9,
            eps_c: float = 10.0, eps_s: float = 10.0) -> Tuple[int, float]:
        r"""
        Run the entire KMC simulation.

        If ``mode`` is ``'dynamic'``, then after every ``niter`` iterations, the step size is
        reduced by ``eps_s`` if the norm of the change in concentrations vector is within ``eps_c``
        times the step size.

        Iteration is stopped when the step size goes below ``tol_s`` (for dynamic mode), or time
        step size goes above ``tol_t`` (for both modes). This is checked every ``niter``
        iterations.

        Parameters
        ----------
        mode : ('static' | 'dynamic'), default='static'
            Whether to use a static or dynamic step size.
        step : float, default=1.0e-6
            Step size for change in concentration at each iteration.
        niter : int, default=1000
            Number of iterations to run before changing step size or stopping.
        maxcall : int, default=1000
            Number of times to call ``self.iterate`` method.
        tol_t : float, default=1.0e12
            Time convergence tolerance.
        tol_s : float, default=1.0e-9
            Step convergence tolerance. Only used for ``mode='dynamic'``.
        eps_c : float, default=10.0
            Decrease the step size when the change in concentrations vector between ``niter``
            iterations is less than this value. Only used for ``mode='dynamic'``.
        eps_s : float, default=10.0
            Divide the step size by this value when it is decreased. Only used for
            ``mode='dynamic'``.

        Returns
        -------
        ncall : int
            Number of times ``self.iterate`` was called.
        step : float
            Final concentration step size.

        """
        ncall = 0
        if mode == 'static':
            while ncall < maxcall:
                # Run ``niter`` iterations
                prev_time = self._time
                self.iterate(step=step, niter=niter)
                ncall += 1
                # Check for time convergence
                if self._time - prev_time > tol_t:
                    break

        elif mode == 'dynamic':
            dconc = np.copy(self._concs)
            while ncall < maxcall:
                # Run ``niter`` iterations
                prev_time = self._time
                self.iterate(step=step, niter=niter)
                ncall += 1
                # Check for time convergence
                if self._time - prev_time > tol_t:
                    break
                # Check for decrease in step size
                dconc -= self._concs
                if np.sqrt(np.dot(dconc, dconc)) < step * eps_c:
                    step /= eps_s
                # Check for step convergence
                if step < tol_s:
                    break
                # Prepare for next iteration
                dconc[:] = self._concs
        else:
            raise ValueError("'mode' must be either 'static' or 'dynamic'")

        return ncall, step


class KMCSolver(BaseKMCSolver):
    r"""
    Kinetic Monte Carlo simultaneous equilibrium solver class.

    Run a Kinetic Monte Carlo (KMC) simulation to find the equilibrium concentrations for a system
    of simultaneous equilibria.

    At every iteration, rates of reaction are updated based on current concentrations. Based on the
    rates, a reaction is selected to occur, where the concentrations of each involved species change
    by the species' stoichiometric coefficient times the chosen concentration step. Each reaction is
    either a forward or reverse reaction.

    """
    _iterate = staticmethod(run_kmc)


class NEKMCSolver(BaseKMCSolver):
    r"""
    Net-Event Kinetic Monte Carlo simultaneous equilibrium solver class.

    Run a Net-Event Kinetic Monte Carlo (NEKMC) simulation to find the equilibrium concentrations
    for a system of simultaneous equilibria.

    At every iteration, net rates of reaction are updated based on current concentrations. Based on
    the net rates, a reaction is selected to occur, where the concentrations of each involved
    species change by the species' stoichiometric coefficient times the chosen concentration step.

    """
    _iterate = staticmethod(run_nekmc)
