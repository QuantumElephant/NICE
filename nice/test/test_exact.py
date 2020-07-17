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

from nice import NEKMCSolver, ExactSolver


def test_raises():
    initial_concs = np.array([1.0, 0.2, 0.4])
    stoich_coeffs = np.array([[-0.5, 1.0, 0.0], [-0.5, -1.0, 1.0]])
    keq_values = np.array([1, 0.1])
    solver = ExactSolver(initial_concs, stoich_coeffs, keq_values=keq_values)
    assert_raises(ValueError, solver.optimize, guess=[1., 1., 1., 1., 1., 1.])
    assert_raises(ValueError, solver.optimize, guess=[1., 1.], mode='invalid')


def test_mol_keq_expressions():
    initial_concs = np.array([1.0, 0.2, 0.4])
    stoich_coeffs = np.array([[-0.5, 1.0, 0.0], [-0.5, -1.0, 1.0]])
    keq_values = np.array([1, 0.1])
    solver = ExactSolver(initial_concs, stoich_coeffs, keq_values=keq_values)
    mol_exps = solver._mol_expressions(np.array([0.45500537, -0.30738121]))
    assert_allclose(mol_exps, [0.9261879203, 0.9623865752, 0.0926187920])
    mol_exps = solver._mol_expressions(np.array([0., 0.]))
    assert_allclose(mol_exps, [1.0, 0.2, 0.4])
    keq_exps = solver._keq_expressions(np.array([0., 0.]))
    assert_allclose(keq_exps, [-0.8, 1.9])
    keq_exps = solver._keq_expressions(np.array([0.1, 0.1]))
    assert_allclose(keq_exps, [-0.7891814893, 2.5352313834])
    keq_exps = solver._keq_expressions(np.array([0.45500537, -0.30738121]))
    assert_allclose(keq_exps, [0.0, 0.0], rtol=0, atol=1e-8)


def test_optimize_newton():
    initial_concs = np.array([1.0, 0.2, 0.4])
    stoich_coeffs = np.array([[-0.5, 1.0, 0.0], [-0.5, -1.0, 1.0]])
    keq_values = np.array([1, 0.1])
    solver = ExactSolver(initial_concs, stoich_coeffs, keq_values=keq_values)
    solver.optimize(guess=[0.1, -0.1], mode='newton')
    assert_allclose(solver.concs, [0.9261879203, 0.9623865752, 0.0926187920])


def test_optimize_bound():
    initial_concs = np.array([1.0, 0.2, 0.4])
    stoich_coeffs = np.array([[-0.5, 1.0, 0.0], [-0.5, -1.0, 1.0]])
    keq_values = np.array([1, 0.1])
    solver = ExactSolver(initial_concs, stoich_coeffs, keq_values=keq_values)
    solver.optimize(guess=[0.1, -0.1], mode='bound', tol=1e-12)
    assert_allclose(solver.concs, [0.9261879203, 0.9623865752, 0.0926187920], rtol=1e-5)


def test_optimize_cma():
    initial_concs = np.array([1.0, 0.2, 0.4])
    stoich_coeffs = np.array([[-0.5, 1.0, 0.0], [-0.5, -1.0, 1.0]])
    keq_values = np.array([1, 0.1])
    solver = ExactSolver(initial_concs, stoich_coeffs, keq_values=keq_values)
    solver.optimize(guess=[0.0, 0.0], mode='cma', tol=1e-12)
    assert_allclose(solver.concs, [0.9261879203, 0.9623865752, 0.0926187920], rtol=1e-5)


def test_optimize_netkmc_guess():
    initial_concs = np.array([1.0, 0.2, 0.4])
    stoich_coeffs = np.array([[-0.5, 1.0, 0.0], [-0.5, -1.0, 1.0]])
    keq_values = np.array([1, 0.1])
    ksolver = NEKMCSolver(initial_concs, stoich_coeffs, keq_values=keq_values)
    ksolver.iterate(step=1e-6, niter=50000)
    esolver = ExactSolver(initial_concs, stoich_coeffs, keq_values=keq_values)
    esolver.optimize(guess=ksolver.compute_zeta())
    assert_allclose(esolver.concs, [0.9261879203, 0.9623865752, 0.0926187920])
