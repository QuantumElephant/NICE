#!/usr/bin/env python

import numpy as np
from nice import *


stoich_coeff = np.array([[-0.5, 1.0, 0.0], [-0.5, -1.0, 1.0]])
initial_concentrations_mc = [10000, 2000, 4000]
initial_concentrations_ex = [1.0, 0.2, 0.4]
keq_values = [1.0, 0.1]
initial_guess = [-0.245, -0.4334] # guess must be chosen fairly carefully!

'''
montecarlo_solver = KMCSolver(initial_concentrations_ex, keq_values, stoich_coeff)
montecarlo_solver.run_simulation(1, net_rate_KMC = True)
'''

exact_solver = ExactEqmSolver(initial_concentrations_ex, keq_values, stoich_coeff, initial_guess)
exact_solver.solve_final_concentrations(keq_mol_frac = False)

