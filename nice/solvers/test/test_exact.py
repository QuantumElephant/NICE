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


def test_mol_molar_expressions():

    solver = ExactEqmSolver(initial_conc, keq, coeff, initial_guess = guess)
    eqns = map(str, solver.setup_mol_molar_expressions()) # Converted to a string due to the sympy symbol objects in the expressions
   
    assert eqns == ['-0.5*z0 - 0.5*z1 + 1.0', '1.0*z0 - 1.0*z1 + 0.2', '1.0*z1 + 0.4']


# This needs to be removed if support for molfractions is removed
def test_molfractions():

    solver = ExactEqmSolver(initial_conc, keq, coeff, initial_guess = guess)
    z0, z1 = sympy.symbols('z0 z1')
    solver.mol_exps = [-0.5*z0 - 0.5*z1 + 1.0, 1.0*z0 - 1.0*z1 + 0.2, 1.0*z1 + 0.4]
    eqns = map(str, solver.get_molfractions())
    
    assert eqns == ['(-0.5*z0 - 0.5*z1 + 1.0)/(0.5*z0 - 0.5*z1 + 1.6)', '(1.0*z0 - 1.0*z1 + 0.2)/(0.5*z0 - 0.5*z1 + 1.6)', '(1.0*z1 + 0.4)/(0.5*z0 - 0.5*z1 + 1.6)']


# Checks the keq expressions generated using the mol fraction expressions
def test_keq_exps_molfrac():

    solver = ExactEqmSolver(initial_conc, keq, coeff, initial_guess = guess)
    z0, z1 = sympy.symbols('z0 z1')
    solver.mol_fractions = [(-0.5*z0 - 0.5*z1 + 1.0)/(0.5*z0 - 0.5*z1 + 1.6), (1.0*z0 - 1.0*z1 + 0.2)/(0.5*z0 - 0.5*z1 + 1.6), (1.0*z1 + 0.4)/(0.5*z0 - 0.5*z1 + 1.6)]
    eqns = map(str, solver.setup_keq_exps_molfrac())
    
    assert eqns == ['((-0.5*z0 - 0.5*z1 + 1.0)/(0.5*z0 - 0.5*z1 + 1.6))**(-0.5)*((1.0*z0 - 1.0*z1 + 0.2)/(0.5*z0 - 0.5*z1 + 1.6))**1.0 - 1', '((1.0*z1 + 0.4)/(0.5*z0 - 0.5*z1 + 1.6))**1.0*((-0.5*z0 - 0.5*z1 + 1.0)/(0.5*z0 - 0.5*z1 + 1.6))**(-0.5)*((1.0*z0 - 1.0*z1 + 0.2)/(0.5*z0 - 0.5*z1 + 1.6))**(-1.0) - 0.1']

  

# Checks the keq expressions generated using the concentration expressions
def test_keq_exps_conc():

    solver = ExactEqmSolver(initial_conc, keq, coeff, initial_guess = guess)
    z0, z1 = sympy.symbols('z0 z1')
    solver.mol_exps = [-0.5*z0 - 0.5*z1 + 1.0, 1.0*z0 - 1.0*z1 + 0.2, 1.0*z1 + 0.4]
    eqns = map(str, solver.setup_keq_exps_conc())
    print 'eqns'
    print eqns 
    assert eqns == ['(-0.5*z0 - 0.5*z1 + 1.0)**(-0.5)*(1.0*z0 - 1.0*z1 + 0.2)**1.0 - 1', '(1.0*z1 + 0.4)**1.0*(-0.5*z0 - 0.5*z1 + 1.0)**(-0.5)*(1.0*z0 - 1.0*z1 + 0.2)**(-1.0) - 0.1']


def test_scipy_func():

    solver = ExactEqmSolver(initial_conc, keq, coeff, initial_guess = guess)
    z0, z1 = sympy.symbols('z0 z1')
    solver.zeta_vars = (z0, z1)
    solver.keq_exps = ['(-0.5*z0 - 0.5*z1 + 1.0)**(-0.5)*(1.0*z0 - 1.0*z1 + 0.2)**1.0 - 1.0', '(1.0*z1 + 0.4)**1.0*(-0.5*z0 - 0.5*z1 + 1.0)**(-0.5)*(1.0*z0 - 1.0*z1 + 0.2)**(-1.0) - 0.1']
    solns = solver.convert_scipy_func(z = [0,0])

    assert solns == [-0.8, 1.9]
    

# This function doesn't test the get_zeta_values code directly- the key statements are copied here to test them.
def test_zeta_values():

    solver = ExactEqmSolver(initial_conc, keq, coeff, initial_guess = [-0.245, -0.4334])    
    z0, z1 = sympy.symbols('z0 z1')
    def keq_conc_func(z):
        return [(-0.5*z[0] - 0.5*z[1] + 1.0)**(-0.5)*(1.0*z[0] - 1.0*z[1] + 0.2)**1.0 - 1.0, (1.0*z[1] + 0.4)**1.0*(-0.5*z[0] - 0.5*z[1] + 1.0)**(-0.5)*(1.0*z[0] - 1.0*z[1] + 0.2)**(-1.0) - 0.1]
    zeta_values = fsolve(keq_conc_func, solver.initial_guess)
    zeta_values = list(zeta_values)
    for i, zeta in enumerate(zeta_values):
        zeta_values[i] = round(zeta, 8)

    assert zeta_values == [0.45500537, -0.30738121]

    def keq_molfrac_func(z):
        return [((-0.5*z[0] - 0.5*z[1] + 1.0)/(0.5*z[0] - 0.5*z[1] + 1.6))**(-0.5)*((1.0*z[0] - 1.0*z[1] + 0.2)/(0.5*z[0] - 0.5*z[1] + 1.6))**1.0 - 1.0, ((1.0*z[1] + 0.4)/(0.5*z[0] - 0.5*z[1] + 1.6))**1.0*((-0.5*z[0] - 0.5*z[1] + 1.0)/(0.5*z[0] - 0.5*z[1] + 1.6))**(-0.5)*((1.0*z[0] - 1.0*z[1] + 0.2)/(0.5*z[0] - 0.5*z[1] + 1.6))**(-1.0) - 0.1]
    zeta_values = fsolve(keq_molfrac_func, solver.initial_guess)
    zeta_values = list(zeta_values)
    for i, zeta in enumerate(zeta_values):
        zeta_values[i] = round(zeta, 8)

    assert zeta_values == [0.76867652, -0.32231793]


def test_concentrations():

    solver = ExactEqmSolver(initial_conc, keq, coeff, initial_guess = guess)
    z0, z1 = sympy.symbols('z0 z1')
    solver.mol_exps = [-0.5*z0 - 0.5*z1 + 1.0, 1.0*z0 - 1.0*z1 + 0.2, 1.0*z1 + 0.4]
    solver.zeta_vars = (z0, z1)
    solver.zeta_values = [ 0.45500537, -0.30738121]
    
    final = solver.get_final_concentrations()
    print final
    assert final == [0.926187920000000, 0.962386580000000, 0.0926187900000000]

# TODO test the run method (solve_final_concentrations())

