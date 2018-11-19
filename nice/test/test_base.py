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
    stoich_coeffs = np.array([[-0.5, 1.0, 0.0], [-0.5, -1.0, 1.0]])
    keq_values = np.array([1, 0.1])
    rate_consts = np.array([[1, 0.1], [0.1, 1]])
    assert_raises(ValueError, BaseSolver, initial_concs, stoich_coeffs)
    assert_raises(ValueError, BaseSolver, 1, stoich_coeffs, keq_values=keq_values)
    assert_raises(ValueError, BaseSolver, initial_concs, 1, keq_values=keq_values)
    assert_raises(ValueError, BaseSolver, initial_concs, stoich_coeffs, keq_values=1)
    assert_raises(ValueError, BaseSolver, initial_concs, stoich_coeffs, rate_consts=1)
    assert_raises(ValueError, BaseSolver, initial_concs, stoich_coeffs, rate_consts=np.ones(3))
    assert_raises(ValueError, BaseSolver, initial_concs, stoich_coeffs, keq_values=keq_values, rate_consts=rate_consts)
    assert_raises(ValueError, BaseSolver, np.zeros((3, 3)), stoich_coeffs, keq_values=keq_values)
    assert_raises(ValueError, BaseSolver, initial_concs, stoich_coeffs, keq_values=np.zeros((3, 3)))
    assert_raises(ValueError, BaseSolver, initial_concs,  np.zeros(3), keq_values=keq_values,)
    assert_raises(ValueError, BaseSolver, initial_concs, np.zeros((3, 3)), keq_values=keq_values)


def test_object_raises():
    initial_concs = np.array([1.0, 0.2, 0.4])
    stoich_coeffs = np.array([[-0.5, 1.0, 0.0], [-0.5, -1.0, 1.0]])
    rate_consts = np.array([[1, 0.1], [0.1, 1]])
    solver = BaseSolver(initial_concs, stoich_coeffs, rate_consts=rate_consts)
    assert_raises(AttributeError, lambda : solver.keq_values)
    assert_raises(NotImplementedError, solver.run)


def test_properties():
    initial_concs = np.array([1.0, 0.2, 0.4])
    stoich_coeffs = np.array([[-0.5, 1.0, 0.0], [-0.5, -1.0, 1.0]])
    keq_values = np.array([1, 0.1])
    solver = BaseSolver(initial_concs, stoich_coeffs, keq_values=keq_values)
    assert_allclose(solver.initial_concs, initial_concs)
    assert_allclose(solver.stoich_coeffs, stoich_coeffs)
    assert_allclose(solver.concs, initial_concs)
    assert_allclose(solver.keq_values, keq_values)
    rate_consts = np.vstack((solver.fwd_rate_consts, solver.rev_rate_consts))
    solver2 = BaseSolver(initial_concs, stoich_coeffs, rate_consts=rate_consts)
    assert_allclose(solver2.fwd_rate_consts, rate_consts[0])
    assert_allclose(solver2.rev_rate_consts, rate_consts[1])
