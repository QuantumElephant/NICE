|Python 2.7| |Python 3.6|

NICE
====

Net-Event Kinetic Monte Carlo and exact solvers for simultaneous
equilibrium equations.

Dependencies
------------

The following libraries are required to run PyCI:

-  NumPy_ (≥1.13.0)
-  SciPy_ (≥0.17.0)
-  Nosetests_ (optional: to run tests)

.. _NumPy: http://numpy.org/
.. _SciPy: http://www.scipy.org/scipylib/index.html
.. _Nosetests: http://nose.readthedocs.io/

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

.. |Python 2.7| image:: http://img.shields.io/badge/python-2.7-blue.svg
   :target: https://docs.python.org/2.7/
.. |Python 3.6| image:: http://img.shields.io/badge/python-3.6-blue.svg
   :target: https://docs.python.org/3.6/
