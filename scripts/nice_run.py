#!/usr/bin/env python

from __future__ import division
import argparse

import numpy as np


def get_filename():

    parser = argparse.ArgumentParser(description= 'Solves coupled equillibria using various algorithms.' )
    parser.add_argument('filename', help = 'The name of your input file.')
    args = parser.parse_args()
    
    return args.filename


def read_input(filename = get_filename()):
    # List of all the 'single line' keywords (excludes the coefficients keyword)
    keyword_list = ['solver', 'initial', 'keq', 'guess', 'step', 'maxiter', 'nekmc']
    parsed_input = {}
    coefficients = []
    with open filename as handle:
        # This loop reads all the data, except for the coefficients
        for i, line in enumerate(handle):
            if line == '\n':
                continue
            elif line.split('=')[0] in parsed_input.keys():
                raise ValueError, 'A repeated keyword was found on line %s.' %(i + 1)
            elif line.split('=')[0] in keyword_list:
                parsed_input[line.split('=')[0]] = line.split('=')[1]
            elif line.split('=')[0] == 'coefficients':
                break
            else:
                raise ValueError, 'An unknown keyword was found on line %s.' %(i + 1)
        # Reads in the coefficient matrix
        for line in handle:
            if line == '\n':
                continue
            else:
                coefficients.append(line.split())

        parsed_input['coefficients'] = np.array(coefficients, dtype = float)


# Stops users from doing dumb stuff.
def check_input(parsed_input):
    
    # Checks if the mandatory arguments are specified
    if 'solver' not in parsed_input.keys():
        raise KeyError, 'The type of solver to be used must be specified.'
    elif 'initial' not in parsed_input.keys():
        raise KeyError, 'The initial concentrations to be used must be specified.'
    elif 'keq' not in parsed_input.keys():
        raise KeyError, 'The equilibrium constants for each reaction must be specified.'

    # Checks if the correct options are specified for each solver
    if parsed_input['solver'] == 'kmc':
        if 'guess' in parsed_input.keys():
            raise ValueError, "Kinetic Monte-Carlo doesn't use an initial guess. Please remove the guess from your input file and try again." 
         # Specifies defaults for optional options. TODO write functions that automatically set decent default values.
        if 'step' not in parsed_input.keys():
            warnings.warn('It is highly reccomended that the step value be manually specified. A default value of 0.0000001 will be used.')
            parsed_input['step'] = 0.0000001
        elif 'maxiter' not in parsed_input.keys():
            warnings.warn('It is highly reccomended that the maxiter value be manually specified. A default value of 100000000 will be used.')
            parsed_input['maxiter'] = 100000000
        elif 'nekmc' not in parsed_input.keys():
            print 'The net-rate KMC (nekmc) keyword is not specified. By default, the solver will use net rates.'
            parsed_input['nekmc'] = True

    elif parsed_input['solver'] == 'exact':
        if 'guess' in parsed_input.keys(): # Allows steps, maxiter and nekmc to be specified if monte carlo is used as a guess.
            if parsed_input['guess'] = 'kmc' 
                if 'step' not in parsed_input.keys():
                    warnings.warn('It is highly reccomended that the kmc step value be manually specified. A default value of 0.0000001 will be used.')
                    parsed_input['step'] = 0.0000001
                elif 'maxiter' not in parsed_input.keys():
                    warnings.warn('It is highly reccomended that the kmc maxiter value be manually specified. A default value of 100000000 will be used.')
                    parsed_input['maxiter'] = 100000000
                elif 'nekmc' not in parsed_input.keys():
                    print 'The net-rate KMC (nekmc) keyword is not specified. By default, the solver will use net rates.'
                    parsed_input['nekmc'] = True
        elif 'guess' not in parsed_input.keys():
            warnings.warn('An initial zeta guess should be specified- convergence issues may occur using the default guess.')
            parsed_input['guess'] = None
            if 'step' in parsed_input.keys():
                raise ValueError, "The exact solver doesn't use a step value. Please remove the step value from your input file and try again." 
            if 'maxiter' in parsed_input.keys():
                raise ValueError, "The exact solver doesn't use a maxiter value. Please remove the maxiter value from your input file and try again." 
            if 'step' in parsed_input.keys():
                raise ValueError, "The exact solver doesn't use a step value. Please remove the step value from your input file and try again."
    else:
        raise ValueError, 'The solver specified is not a valid option. Please read the documentation for more info.'
           
    # Checks to ensure that the user passes the right type of input for each argument.

    if not isinstance(parsed_input['initial'], list):
        raise ValueError, 'Please ensure that the initial concentrations list is in the correct format.'

    if not isinstance(parsed_input['keq'], list):
        raise ValueError, 'Please ensure your equillibrium constant list is in the correct format.'

    if not isinstance(parsed_input['coefficients'], np.ndarray):
        try:
            parsed_input['coefficients'] = np.array(parsed_input['coefficients'])
        except:
            raise ValueError, 'Please check your coefficient matrix- could not convert to an array.'
    
    if not isinstance(parsed_input['step'], float):
        raise ValueError, 'Please ensure that the concentration_step value is a float value.'
    elif parsed_input['step'] <= 0:
        raise ValueError, 'The step value must be a positive float value greater than 0.'

    if not isinstance(parsed_input['nekmc'], bool):
        raise ValueError, 'The value of nekmc must be a boolean value (True/False, case sensitive).'

    if not isinstance(parsed_input['maxiter'], int):
        raise ValueError, 'The value of maxiter should be an integer.'
    elif parsed_input['maxiter'] <= 0:
        raise ValueError, 'The value of maxiter should be positive.'

    if not isinstance(parsed_input['guess'], list):
        if not isinstance(parsed_input['guess'], None):
            if parsed_input['guess'] != 'kmc'
                raise ValueError 'YOur initial guess should be either a list, None, or generated from the Kinetic Monte-Carlo solver.'

           
def run_calculation(parsed_input):

    if parsed_input['guess'] == 'kmc':
        montecarlo_solver = KMCSolver(parsed_input['initial'], parsed_input['keq'], parsed_input['coefficients'], concentration_step = parsed_input['step'])
        montecarlo_solver.run_simulation(parsed_input['maxiter'], net_rate_KMC = parsed_input['nekmc'])
        bridge = BridgeSolvers(parsed_input['initial'], montecarlo_solver.concentrations, parsed_input['coefficients']) 
        parsed_input['guess'] = bridge.get_zeta_guess()

    if parsed_input['solver'] == 'kmc':
        montecarlo_solver = KMCSolver(parsed_input['coefficients'], parsed_input['initial'], parsed_input['keq'], concentration_step = parsed_input['step'])
        montecarlo_solver.run_simulation(parsed_input['maxiter'], net_rate_KMC = parsed_input['nekmc'])
    elif parsed_input['solver'] == 'exact':
        exact_solver = ExactEqmSolver(parsed_input['initial'], parsed_input['keq'], parsed_input['coefficients'], parsed_input['initial_guess'])

