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
from numpy.testing import assert_raises, assert_allclose

from nice import NEKMCSolver


def test_raises():
    initial_concs = np.array([1.0, 0.2, 0.4])
    stoich_coeffs = np.array([[-0.5, 1.0, 0.0], [-0.5, -1.0, 1.0]])
    keq_values = np.array([1, 0.1])
    solver = NEKMCSolver(initial_concs, stoich_coeffs, keq_values=keq_values, phi=1.0)
    assert_raises(ValueError, solver.run_simulation, mode='invalid')


def test_properties():
    initial_concs = np.array([1.0, 0.2, 0.4])
    stoich_coeffs = np.array([[-0.5, 1.0, 0.0], [-0.5, -1.0, 1.0]])
    keq_values = np.array([1, 0.1])
    solver = NEKMCSolver(initial_concs, stoich_coeffs, keq_values=keq_values, phi=1.0)
    assert_allclose(solver.fwd_rate_consts, [0.5, 0.0909090909])
    assert_allclose(solver.rev_rate_consts, [0.5, 0.9090909090])
    assert_allclose(solver.fwd_rates, [0.5, 0.0181818181])
    assert_allclose(solver.rev_rates, [0.1, 0.3636363636])
    assert_allclose(solver.net_rates, [0.4, -0.3454545454])


def test_run_static_1():
    initial_concs = np.array([1.0, 0.2, 0.4])
    stoich_coeffs = np.array([[-0.5, 1.0, 0.0], [-0.5, -1.0, 1.0]])
    keq_values = np.array([1, 0.1])
    solver = NEKMCSolver(initial_concs, stoich_coeffs, keq_values=keq_values, phi=1.0)
    solver.run_simulation(mode='static', step=1e-7, maxcall=10000)
    assert_allclose(solver.concs, [0.9261879203, 0.9623865752, 0.0926187920], rtol=1e-5)


def test_run_dynamic_1():
    initial_concs = np.array([1.0, 0.2, 0.4])
    stoich_coeffs = np.array([[-0.5, 1.0, 0.0], [-0.5, -1.0, 1.0]])
    keq_values = np.array([1, 0.1])
    solver = NEKMCSolver(initial_concs, stoich_coeffs, keq_values=keq_values, phi=1.0)
    solver.run_simulation(mode='dynamic', step=1e-4, niter=1000, maxcall=10000, tol_s=1e-9)
    assert_allclose(solver.concs, [0.9261879203, 0.9623865752, 0.0926187920], rtol=1e-5)


def test_run_static_2():
    initial_concs = np.array([1.0, 0.1, 1e-4, 0.0, 0.0, 0.0])
    stoich_coeffs = np.array([[-1, -1,  0,  1,  0,  0],
                              [ 0, -1, -1,  0,  1,  0],
                              [ 0,  0, -1, -1,  0,  1],
                              [-1,  0,  0,  0, -1,  1]])
    keq_values = np.array([1e7, 1e9, 1e7, 1e9])
    solver = NEKMCSolver(initial_concs, stoich_coeffs, keq_values=keq_values, phi=1.0)
    solver.run_simulation(mode='static', step=1e-8, maxcall=50000)
    result = [9.0e-01, 1.10212630e-08, 1.00002433e-10, 9.98999891e-02, 0.0, 9.99999e-05]
    assert_allclose(solver.concs, result, atol=1e-7, rtol=0)


def test_run_dynamic_2():
    initial_concs = np.array([1.0, 0.1, 1e-4, 0.0, 0.0, 0.0])
    stoich_coeffs = np.array([[-1, -1,  0,  1,  0,  0],
                              [ 0, -1, -1,  0,  1,  0],
                              [ 0,  0, -1, -1,  0,  1],
                              [-1,  0,  0,  0, -1,  1]])
    keq_values = np.array([1e7, 1e9, 1e7, 1e9])
    solver = NEKMCSolver(initial_concs, stoich_coeffs, keq_values=keq_values, phi=1.0)
    solver.run_simulation(mode='dynamic', step=1e-6, niter=1000, maxcall=10000, tol_s=1e-12)
    result = [9.0e-01, 1.10212630e-08, 1.00002433e-10, 9.98999891e-02, 0.0, 9.99999e-05]
    assert_allclose(solver.concs, result, atol=1e-7, rtol=0)
