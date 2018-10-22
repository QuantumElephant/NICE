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


import numpy as np
from numpy.testing import assert_raises, assert_allclose

from nice.base import BaseSolver


def test_init_raises():
    initial_concs = np.array([1.0, 0.2, 0.4])
    keq_values = np.array([1, 0.1])
    stoich_coeffs = np.array([[-0.5, 1.0, 0.0], [-0.5, -1.0, 1.0]])
    assert_raises(ValueError, BaseSolver, 1, keq_values, stoich_coeffs)
    assert_raises(ValueError, BaseSolver, initial_concs, 1, stoich_coeffs)
    assert_raises(ValueError, BaseSolver, initial_concs, keq_values, 1)
    assert_raises(ValueError, BaseSolver, np.zeros((3, 3)), keq_values, stoich_coeffs)
    assert_raises(ValueError, BaseSolver, initial_concs, np.zeros((3, 3)), stoich_coeffs)
    assert_raises(ValueError, BaseSolver, initial_concs, keq_values, np.zeros(3))
    assert_raises(ValueError, BaseSolver, initial_concs, keq_values, np.zeros((3, 3)))


def test_run_raises_notimplemented():
    initial_concs = np.array([1.0, 0.2, 0.4])
    keq_values = np.array([1, 0.1])
    stoich_coeffs = np.array([[-0.5, 1.0, 0.0], [-0.5, -1.0, 1.0]])
    solver = BaseSolver(initial_concs, keq_values, stoich_coeffs)
    assert_raises(NotImplementedError, solver.run)


def test_properties():
    initial_concs = np.array([1.0, 0.2, 0.4])
    keq_values = np.array([1, 0.1])
    stoich_coeffs = np.array([[-0.5, 1.0, 0.0], [-0.5, -1.0, 1.0]])
    solver = BaseSolver(initial_concs, keq_values, stoich_coeffs)
    assert_allclose(solver.initial_concs, initial_concs)
    assert_allclose(solver.keq_values, keq_values)
    assert_allclose(solver.stoich_coeffs, stoich_coeffs)
    assert_allclose(solver.concs, initial_concs)
