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
    'KMCSolver',
    ]


class NEKMCSolver(BaseSolver):
    """
    Net Event Kinetic Monte Carlo simultaneous equilibrium solver class.

    Run a Net-Event Kinetic Monte Carlo (NEKMC) simulation to find the
    equilibrium concentrations for a system of simultaneous equilibria.

    At every iteration, net rates of reaction are updated based on current
    concentrations. Based on the net rates, a reaction is selected to occur,
    where the concentrations of each involved species change by the species'
    stoichiometric coefficient times the chosen concentration step.

    """

    _update_rates = staticmethod(_nekmc.update_rates)
    _run = staticmethod(_nekmc.run_nekmc)

    def __init__(self, initial_concs, stoich_coeffs, keq_values=None, rate_consts=None, phi=1.0):
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
            constants (:math:`phi = k_{+1} + k_{-1}`).

        """
        # Initialize superclass
        super(NEKMCSolver, self).__init__(initial_concs, stoich_coeffs,
                                          keq_values=keq_values,
                                          rate_consts=rate_consts,
                                          phi=phi)
        # Initialize reaction rate arrays
        self._fwd_rates = np.zeros_like(self._fwd_consts)
        self._rev_rates = np.zeros_like(self._fwd_consts)
        self._net_rates = np.zeros_like(self._fwd_consts)
        # Compute forward/reverse/net rates
        nreaction, nspecies = self._stoich_coeffs.shape
        self._update_rates(nspecies, nreaction,
                           self._concs, self._stoich_coeffs.T,
                           self._fwd_consts, self._rev_consts,
                           self._fwd_rates, self._rev_rates, self._net_rates)

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

    def run(self, mode='static', step=1.0e-6, maxiter=100000, inner=1000,
            tol_t=1.0e12, tol_s=1.0e-9, eps_c=10.0, eps_s=10.0):
        """
        Run the (NE)KMC simulation.

        If ``mode`` is ``'dynamic'``, then after every ``inner`` iterations, the
        step size is reduced by ``eps_s`` if the norm of the change in
        concentrations vector is within ``eps_c`` times the step size.

        Iteration is stopped when the step size goes below ``tol_s`` (for
        dynamic mode), or time step size goes above ``tol_t`` (for both modes).
        This is checked every ``inner`` iterations.

        Parameters
        ----------
        mode : ('static' | 'dynamic'), default='static'
            Whether to use a static or dynamic step size.
        step : float, default=1.0e-6
            Step size for change in concentration at each iteration.
        maxiter : int, default=50000
            Number of iterations to run.
        inner : int, default=500
            Number of iterations to run before changing step size or stopping.
        tol_t : float, default=1.0e12
            Time convergence tolerance.
        tol_s : float, default=1.0e-9
            Step convergence tolerance.
            Only used for ``mode='dynamic'``.
        eps_c : float, default=10.0
            Decrease the step size when the change in concentrations vector
            between ``inner`` iterations is less than this value.
            Only used for ``mode='dynamic'``.
        eps_s : float, default=10.0
            Divide the step size by this value when it is decreased.
            Only used for ``mode='dynamic'``.

        Returns
        -------
        time : float
            Reaction time elapsed.
        step : float
            Final concentration step size.
        niter : int
            Number of iterations run.

        """
        nreaction, nspecies = self._stoich_coeffs.shape
        niter = 0
        time = 0.0
        # Check mode
        mode = mode.lower()
        if mode == 'static':
            # Begin iterating
            while niter < maxiter:
                # Run ``inner`` iterations
                dtime = self._run(nspecies, nreaction,
                                  self._concs, self._stoich_coeffs.T,
                                  self._fwd_consts, self._rev_consts,
                                  self._fwd_rates, self._rev_rates, self._net_rates,
                                  step, inner)[3]
                time += dtime
                # Check for time convergence
                if dtime > tol_t:
                    break
                # Prepare for next iteration
                niter += inner
        elif mode == 'dynamic':
            dconc = np.copy(self._concs)
            # Begin iterating
            while niter < maxiter:
                # Run ``inner`` iterations
                dtime = self._run(nspecies, nreaction,
                                  self._concs, self._stoich_coeffs.T,
                                  self._fwd_consts, self._rev_consts,
                                  self._fwd_rates, self._rev_rates, self._net_rates,
                                  step, inner)[3]
                time += dtime
                # Check for time convergence
                if dtime > tol_t:
                    break
                # Compute differences in concentrations
                dconc -= self._concs
                # Check for decrease in step size
                if np.linalg.norm(dconc) < step * eps_c:
                    step /= eps_s
                # Check for step convergence
                if step < tol_s:
                    break
                # Prepare for next iteration
                dconc[:] = self._concs
                niter += inner
        else:
            raise ValueError("'mode' must be either 'static' or 'dynamic'")
        return time, step, niter

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
        return np.linalg.lstsq(self._stoich_coeffs.T, diff, rcond=-1)[0]


class KMCSolver(NEKMCSolver):
    """
    Kinetic Monte Carlo simultaneous equilibrium solver class.

    Run a Kinetic Monte Carlo (KMC) simulation to find the equilibrium
    concentrations for a system of simultaneous equilibria.

    At every iteration, rates of reaction are updated based on current
    concentrations. Based on the rates, a reaction is selected to occur,
    where the concentrations of each involved species change by the
    species' stoichiometric coefficient times the chosen concentration step.
    Each reaction is either a forward or reverse reaction.

    """

    _update_rates = staticmethod(_nekmc.update_rates)
    _run = staticmethod(_nekmc.run_kmc)
