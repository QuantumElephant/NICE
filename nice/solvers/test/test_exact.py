#!/usr/bin/env python

from nice import *

import nose

import sympy
import numpy as np

# The values below are from the phys chem textbook example, pg. 440 ---> Citation?
# The initial values are assumed to be in units of molar instead of mol.


initial_conc = [1.0, 0.2, 0.4]
keq = [1, 0.1]
coeff = np.array([[-0.5, 1.0, 0.0], [-0.5, -1.0, 1.0]])


# Default values already given to the class- provided here to make them more explicit.

mol_frac = False
guess = None


def test_keq_expressions():

    solver = ExactEqmSolver(initial_conc, keq, coeff, initial_guess = guess)
    
    keq_values = solver.setup_keq_expressions(z = [0.45500537, -0.30738121], return_value = 'mol_exps')
    assert np.allclose(keq_values, [0.9261879203, 0.9623865752, 0.0926187920])

    keq_values = solver.setup_keq_expressions(z = [0.0, 0.0], return_value = 'mol_exps')
    assert np.allclose(keq_values, [1.0, 0.2, 0.4])

    keq_values = solver.setup_keq_expressions(z = [0.0,0.0])    
    assert np.allclose(keq_values, [-0.8, 1.9])

    keq_values = solver.setup_keq_expressions(z = [0.1,0.1])    
    assert np.allclose(keq_values, [-0.7891814893, 2.5352313834])
    
    keq_values = solver.setup_keq_expressions(z = [0.45500537, -0.30738121])    
    assert np.allclose(keq_values, [0.0, 0.0])


def test_zeta_values():

    solver = ExactEqmSolver(initial_conc, keq, coeff, initial_guess = [-0.245, -0.4334])

    # Used to overwrite normal equation setup function passed to get_zeta_values
    def setup_test_expressions(z):
        return [(-0.5*z[0] - 0.5*z[1] + 1.0)**(-0.5)*(1.0*z[0] - 1.0*z[1] + 0.2)**1.0 - 1.0, (1.0*z[1] + 0.4)**1.0*(-0.5*z[0] - 0.5*z[1] + 1.0)**(-0.5)*(1.0*z[0] - 1.0*z[1] + 0.2)**(-1.0) - 0.1]

    solver.setup_keq_expressions = setup_test_expressions
    zeta_values = solver.get_zeta_values()

    assert np.allclose(zeta_values, [0.45500537, -0.30738121])


def test_concentrations():

    solver = ExactEqmSolver(initial_conc, keq, coeff, initial_guess = guess)
    
    # Used to overwrite the get_zeta_values method to control passed zeta values.
    def test_zeta():
        return [0.45500537, -0.30738121]

    solver.get_zeta_values = test_zeta
    final_concentrations = solver.solve_final_concentrations()

    assert np.allclose(final_concentrations, [0.9261879203, 0.9623865752, 0.0926187920])
   

