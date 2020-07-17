..
    : Copyright (C) 2020 Ayers Lab.
    :
    : This file is part of NICE.
    :
    : NICE is free software; you can redistribute it and/or modify it under
    : the terms of the GNU General Public License as published by the Free
    : Software Foundation; either version 3 of the License, or (at your
    : option) any later version.
    :
    : NICE is distributed in the hope that it will be useful, but WITHOUT
    : ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    : FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
    : for more details.
    :
    : You should have received a copy of the GNU General Public License
    : along with this program; if not, see <http://www.gnu.org/licenses/>.

|Python 3.8| |Build Status|

NICE
====

(Net-Event) Kinetic Monte Carlo and exact solvers for simultaneous equilibria.

Dependencies
------------

The following programs/libraries are required to build NICE:

-  GFortran_ (or another Fortran compiler, e.g. Flang, PGF, Intel)
-  NumPy_ (including C headers)

The following libraries are required to run NICE:

-  NumPy_
-  SciPy_
-  CMA-ES_ (optional: to use "cma" mode with exact solver)

The following programs/libraries are required to build the NICE documentation:

-  Sphinx_
-  `Read the Docs Sphinx Theme`__

__ Sphinx-RTD-Theme_

Installation
------------

To download NICE via git:

.. code::

    git clone https://github.com/quantumelephant/nice.git
    cd nice

To build NICE in-place for use in cloned folder:

.. code::

    python setup.py build_ext --inplace

To install NICE for a user (i.e., locally):

.. code::

    python setup.py install --user

Building Documentation
----------------------

To install the Read the Docs Sphinx theme via pip:

.. code:: shell

    pip install sphinx-rtd-theme --user

Then, after installing NICE, run the following to build the HTML documentation:

.. code:: shell

    cd doc && make html

Example
-------

We have a set of simultaneous equilibrium equations

.. code::

    1 A        <-->  2 B    K_eq = 1.0
    1 A + 2 B  <-->  2 C    K_eq = 0.1

and initial concentrations

.. code::

    [A] = 1.0 M,  [B] = 0.2 M,  [C] = 0.4 M

We set up the NEKMC solver:

.. code::

    import numpy as np
    import nice

    concs  = np.array([1.0, 0.2, 0.4])
    stoich = np.array([[-0.5, 1.0, 0.0], [-0.5, -1.0, 1.0]])
    keqs   = np.array([1.0, 0.1])
    nekmc  = nice.NEKMCSolver(concs, stoich, keq_values=keqs)

Each row of ``stoich`` represents a reaction. Each column is the stoichiometric coefficient of a
species; the value is negative for a reactant, positive for a product, and zero if the species does
not participate.

It is also possible to initialize the solver by passing the forward and reverse rate constants
directly:

.. code::

    concs  = np.array([1.0, 0.2, 0.4])
    stoich = np.array([[-0.5, 1.0, 0.0], [-0.5, -1.0, 1.0]])
    consts = np.array([[1.0, 0.1],  # forward rate constants
                       [0.1, 1.0]]) # reverse rate constants
    nekmc  = nice.NEKMCSolver(concs, stoich, rate_consts=consts)

We run the solver with concentration step-size matching our desired precision for 50,000 iterations:

.. code::

    nekmc.iterate(step=1e-6, niter=50000)

Finally, we set up and run the exact solver, with initial guess generated from the NEKMC solver, to
get a more precise result:

.. code::

    exact = nice.ExactSolver(concs, stoich, keq_values=keqs)
    exact.optimize(guess=nekmc.compute_zeta(), tol=1e-9)

Now our system is fully converged to equilibrium. Final concentrations for each solver are located
at ``nekmc.concs`` and ``exact.concs``. See the code documentation for more detailed information on
running NICE.

.. _GFortran: http://gcc.gnu.org/wiki/GFortran
.. _NumPy: http://numpy.org/
.. _SciPy: http://www.scipy.org/scipylib/index.html
.. _CMA-ES: http://github.com/CMA-ES/pycma
.. _Sphinx:             http://sphinx-doc.org/
.. _Sphinx-RTD-Theme:   http://sphinx-rtd-theme.readthedocs.io/

.. |Python 3.8| image:: http://img.shields.io/badge/python-3.8-blue.svg
   :target: http://docs.python.org/3.8/
.. |Build Status| image:: http://travis-ci.com/QuantumElephant/NICE.svg?token=cXv5xZ8ji4xAnkUvpsev&branch=master
   :target: http://travis-ci.com/QuantumElephant/NICE
