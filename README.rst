|Python 2.7| |Python 3.6| |Build Status|

NICE
====

Net-Event Kinetic Monte Carlo and exact solvers for simultaneous
equilibrium equations.

Dependencies
------------

The following libraries are required to build PyCI:

-  NumPy_ (≥1.13.0, including system headers)
-  GFortran_ (or another Fortran compiler)

The following libraries are required to run PyCI:

-  NumPy_ (≥1.13.0)
-  SciPy_ (≥0.17.0)
-  Nosetests_ (optional: to run tests)

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

    python setup.py build_ext
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
    keqs   = np.array([1.0, 0.1])
    stoich = np.array([[-0.5, 1.0, 0.0], [-0.5, -1.0, 1.0]])
    nekmc  = nice.NEKMCSolver(concs, keqs, stoich)

Each row of ``stoich`` represents a reaction. Each column is the stoichiometric
coefficient of a species; the value is negative for a reactant, positive for a
product, and zero if the species does not participate.

We run the solver with concentration step-size matching our desired precision
for 50,000 iterations:

.. code::

    nekmc.run(step=1e-6, maxiter=50000)

Finally, we set up and run the exact solver, with initial guess generated from
the NEKMC solver, to get a more precise result:

.. code::

    exact = nice.ExactSolver(concs, keqs, stoich)
    exact.run(guess=nekmc.compute_zeta(), tol=1e-9)

Now our system is fully converged to equilibrium. Final concentrations for each
solver are located at ``nekmc.concs`` and ``exact.concs``. See the code
documentation for more detailed information on running NICE.

.. _NumPy: http://numpy.org/
.. _SciPy: http://www.scipy.org/scipylib/index.html
.. _GFortran: https://gcc.gnu.org/wiki/GFortran
.. _Nosetests: http://nose.readthedocs.io/

.. |Python 2.7| image:: http://img.shields.io/badge/python-2.7-blue.svg
   :target: https://docs.python.org/2.7/
.. |Python 3.6| image:: http://img.shields.io/badge/python-3.6-blue.svg
   :target: https://docs.python.org/3.6/
.. |Build Status| image:: https://travis-ci.com/QuantumElephant/NICE.svg?token=cXv5xZ8ji4xAnkUvpsev&branch=master
   :target: https://travis-ci.com/QuantumElephant/NICE
