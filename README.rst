|Python 3.8| |Build Status|

NICE
====

(Net-Event) Kinetic Monte Carlo and exact solvers for simultaneous equilibria.

Dependencies
------------

The following libraries are required to build NICE:

-  NumPy_ (≥1.13.0, including system headers)
-  GFortran_ (or another Fortran compiler)

The following libraries are required to run NICE:

-  NumPy_ (≥1.13.0)
-  SciPy_ (≥0.17.0)
-  CMA-ES_ (≥2.6.0, optional: to use "cma" mode with exact solver)

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

.. _NumPy: http://numpy.org/
.. _SciPy: http://www.scipy.org/scipylib/index.html
.. _CMA-ES: http://github.com/CMA-ES/pycma
.. _GFortran: http://gcc.gnu.org/wiki/GFortran

.. |Python 3.8| image:: http://img.shields.io/badge/python-3.8-blue.svg
   :target: http://docs.python.org/3.8/
.. |Build Status| image:: http://travis-ci.com/QuantumElephant/NICE.svg?token=cXv5xZ8ji4xAnkUvpsev&branch=master
   :target: http://travis-ci.com/QuantumElephant/NICE
