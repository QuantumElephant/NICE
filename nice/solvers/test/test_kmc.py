#!/usr/bin/env python

# TODO test the run method (solve_final_concentrations())
# TODO: float assertions with rounding (due to numerical error)

from nice import *


def get_rxn():
    '''
    Return the specifications of a set of reactions.
    '''
    # The values below are from the phys chem textbook example, pg. 440 ---> Citation needed?
    # The initial value units are assumed to be molar instead of mol.
    initial_conc = np.array([1.0, 0.2, 0.4])
    keq_values = np.array([1, 0.1])
    stoich_coeff = np.array([[-0.5, 1.0, 0.0], [-0.5, -1.0, 1.0]])
    # These are the default values already given to the class. They are provided here to make things more explicit.
    phi = 1
    step = 0.0000001
    return initial_conc, keq_values, stoich_coeff, phi, step


def test_kmcsolver():
    # Get the rxn data
    initial_conc, keq_values, stoich_coeff, phi, step = get_rxn()
    # KMC simulation using the forward/reverse reaction rates
    solver = KMCSolver(initial_conc, keq_values, stoich_coeff, phi=phi, concentration_step=step, net_rxn=False)
    # check the rate constants
    assert np.allclose(solver.forward_rate_consts, [0.5, 0.0909090909])
    assert np.allclose(solver.reverse_rate_consts, [0.5, 0.9090909090])
    assert len(solver.forward_rate_consts) == len(keq_values)
    assert len(solver.reverse_rate_consts) == len(keq_values)
    assert len(solver.forward_rate_consts) == len(solver.reverse_rate_consts)
    # check the rates
    assert np.allclose(solver.forward_rates, [0.5, 0.0181818181])
    assert np.allclose(solver.reverse_rates, [0.1, 0.3636363636])
    assert len(solver.forward_rates) == len(keq_values)
    assert len(solver.reverse_rates) == len(keq_values)
    assert len(solver.forward_rates) == len(solver.forward_rate_consts)
    assert len(solver.reverse_rates) == len(solver.reverse_rate_consts)
    assert len(solver.forward_rates) == len(solver.reverse_rates)
    # check the rate probability vector
    prob_vector = solver.create_probability_vector()
    assert np.allclose(prob_vector, np.array([0.5092592592, 0.5277777777, 0.6296296296, 1.0]))
    assert len(prob_vector) == 2*len(keq_values)
    assert len(prob_vector) == len(solver.forward_rates) + len (solver.reverse_rates)
    # check select reaction
    selected_rxn = solver.select_reaction(rate=0.0)
    assert selected_rxn == 0
    selected_rxn = solver.select_reaction(rate=0.6000)
    assert selected_rxn == 2
    selected_rxn = solver.select_reaction(rate=0.9)
    assert selected_rxn == 3
    # check do reaction
    solver.do_reaction(rxn_index=0)
    assert np.allclose(solver.concentrations, np.array([0.99999995, 0.2000001, 0.4]))
    solver.do_reaction(rxn_index=3)  #TODO:changing the index to other values wouldn't change
    assert np.allclose(solver.concentrations, np.array([1.0, 0.2000002, 0.3999999]))


def test_kmcsolver_net():
    # Get the rxn data
    initial_conc, keq_values, stoich_coeff, phi, step = get_rxn()
    # KMC simulation using the net reaction rates
    solver = KMCSolver(initial_conc, keq_values, stoich_coeff, phi=phi, concentration_step=step, net_rxn=True)
    # check the rate constants
    assert np.allclose(solver.forward_rate_consts, [0.5, 0.0909090909])
    assert np.allclose(solver.reverse_rate_consts, [0.5, 0.9090909090])
    # check the rates
    assert np.allclose(solver.forward_rates, [0.5, 0.0181818181])
    assert np.allclose(solver.reverse_rates, [0.1, 0.3636363636])
    assert np.allclose(solver.net_rates, [0.4, -0.3454545454])
    assert np.allclose(solver.net_rates, [0.4, -0.3454545454])
    assert len(solver.net_rates) == len(keq_values)
    assert len(solver.net_rates) == len(solver.forward_rates)
    assert len(solver.net_rates) == len(solver.reverse_rates)
    # check the rate probability vector
    prob_vector = solver.create_probability_vector()
    assert np.allclose(prob_vector, [0.5365853658, 1.0])
    assert len(prob_vector) == len(keq_values)
    # check do reaction
    solver.do_reaction(rxn_index=0)
    assert np.allclose(solver.concentrations, np.array([0.99999995, 0.2000003, 0.3999999]))
    solver.do_reaction(rxn_index=1)
    assert np.allclose(solver.concentrations, np.array([1.0, 0.2000004, 0.3999998]))
