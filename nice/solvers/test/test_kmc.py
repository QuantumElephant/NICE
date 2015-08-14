#!/usr/bin/env python

# TODO test the run method (solve_final_concentrations())

from nice import *

import nose
import numpy as np

# TODO: float assertions with rounding (due to numerical error)

'''
The values below are from the phys chem textbook example, pg. 440 ---> Citation needed?
The initial value units are assumed to be molar instead of mol.
'''

initial_conc = [1.0, 0.2, 0.4]
keq_values = [1, 0.1]
stoich_coeff = np.array([[-0.5, 1.0, 0.0], [-0.5, -1.0, 1.0]])

'''
These are the default values already given to the class. They are provided here to make things more explicit.
'''
phi = 1
step = 0.0000001


def test_rate_constants():

    solver = KMCSolver(initial_conc, keq_values, stoich_coeff, phi= phi, concentration_step = step)
    forward_rate_consts, reverse_rate_consts = solver.get_rate_constants()

    assert np.allclose(forward_rate_consts, [0.5, 0.0909090909])
    assert np.allclose(reverse_rate_consts, [0.5, 0.9090909090])
    assert len(forward_rate_consts) == len(keq_values)
    assert len(reverse_rate_consts) == len(keq_values)
    assert len(forward_rate_consts) == len(reverse_rate_consts)


def test_rates():

    solver = KMCSolver(initial_conc, keq_values, stoich_coeff, phi= phi, concentration_step = step)
    solver.forward_rate_consts = [0.5, 0.09090909090909094]
    solver.reverse_rate_consts = [0.5, 0.9090909090909091]
    forward_rates, reverse_rates = solver.get_rates()
 
    assert np.allclose(forward_rates, [0.5, 0.0181818181])
    assert np.allclose(reverse_rates, [0.1, 0.3636363636])
    assert len(forward_rates) == len(keq_values)
    assert len(reverse_rates) == len(keq_values)
    assert len(forward_rates) == len(solver.forward_rate_consts)
    assert len(reverse_rates) == len(solver.reverse_rate_consts)
    assert len(forward_rates) == len(reverse_rates)


def test_net_rates():

    solver = KMCSolver(initial_conc, keq_values, stoich_coeff, phi= phi, concentration_step = step)
    solver.forward_rates = [0.5, 0.018181818181818188]
    solver.reverse_rates = [0.1, 0.36363636363636365]
    net_rates = solver.get_net_rates()

    assert np.allclose(net_rates, [0.4, -0.3454545454])
    assert len(net_rates) == len(keq_values)
    assert len(net_rates) == len(solver.forward_rates)
    assert len(net_rates) == len(solver.reverse_rates)


def test_probability_vector():

    solver = KMCSolver(initial_conc, keq_values, stoich_coeff, phi= phi, concentration_step = step)
    solver.forward_rates = [0.5, 0.018181818181818188]
    solver.reverse_rates = [0.1, 0.36363636363636365]
    prob_vector = solver.create_probability_vector()
    
    assert np.allclose(prob_vector, [0.5092592592, 0.5277777777, 0.6296296296, 1.0])
    assert len(prob_vector) == 2*len(keq_values)
    assert len(prob_vector) == len(solver.forward_rates) + len (solver.reverse_rates)


def test_net_probability_vector():

    solver = KMCSolver(initial_conc, keq_values, stoich_coeff, phi= phi, concentration_step = step)
    solver.net_rates = [0.4, -0.34545454545454546]
    net_prob_vector = solver.create_net_rate_probability_vector()
    
    assert np.allclose(net_prob_vector, [0.5365853658, 1.0])
    assert len(net_prob_vector) == len(keq_values)


def test_random_value():

    solver = KMCSolver(initial_conc, keq_values, stoich_coeff, phi= phi, concentration_step = step)
    r = solver.select_random_value()
    assert 0.0 <= r <= 1.0


def test_select_reaction():

    solver = KMCSolver(initial_conc, keq_values, stoich_coeff, phi= phi, concentration_step = step)
    solver.r = 0.0
    solver.probability_vector = [0.5092592592592593, 0.52777777777777779, 0.62962962962962965, 1.0]
    selected_rxn = solver.select_reaction()
    
    assert selected_rxn == 0
    
    solver.r = 0.6000
    selected_rxn = solver.select_reaction()    

    assert selected_rxn == 2

    solver.r = 0.9
    selected_rxn = solver.select_reaction()

    assert selected_rxn == 3

    solver.probability_vector = [0.53658536585365857, 1.0]
    solver.r = 0.0
    selected_rxn = solver.select_reaction()
    
    assert selected_rxn == 0

    solver.r = 0.9
    selected_rxn = solver.select_reaction()
    
    assert selected_rxn == 1


def test_reaction():

    solver = KMCSolver(initial_conc, keq_values, stoich_coeff, phi= phi, concentration_step = step)
    solver.probability_vector = [0.5092592592592593, 0.52777777777777779, 0.62962962962962965, 1.0]
    solver.selected_rxn = 0
    solver.do_reaction()
    
    assert np.allclose(solver.concentrations, [0.99999995, 0.2000001, 0.4])
    
    solver = KMCSolver(initial_conc, keq_values, stoich_coeff, phi= phi, concentration_step = step)
    solver.probability_vector = [0.5092592592592593, 0.52777777777777779, 0.62962962962962965, 1.0]
    solver.selected_rxn = 3
    solver.do_reaction()

    assert np.allclose(solver.concentrations, [1.0, 0.2000002, 0.3999999])


def test_net_reaction():

    solver = KMCSolver(initial_conc, keq_values, stoich_coeff, phi= phi, concentration_step = step)
    solver.selected_rxn = 0
    solver.net_rates = [0.4, -0.34545454545454546]
    solver.do_net_reaction()
    
    assert np.allclose(solver.concentrations, [0.99999995, 0.2000003, 0.3999999])

    solver = KMCSolver(initial_conc, keq_values, stoich_coeff, phi= phi, concentration_step = step)
    solver.selected_rxn = 1
    solver.net_rates = [0.4, -0.34545454545454546]
    solver.do_net_reaction()
    print solver.concentrations
    assert np.allclose(solver.concentrations, [1.0, 0.2000004, 0.3999998])


    
