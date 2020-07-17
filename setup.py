# Copyright (C) 2020 Ayers Lab.
#
# This file is part of NICE.
#
# NICE is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free
# Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# NICE is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
# for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>.

from numpy.distutils.core import setup, Extension


NAME = 'NICE'

LICENSE = 'GPLv3'

AUTHOR = 'Ayers Lab'

EMAIL = 'ayers@mcmaster.ca'

DESCRIPTION = '(Net-Event) Kinetic Monte Carlo and exact solvers for simultaneous equilibria'

LONG_DESCRIPTION = open('README.rst', mode='r', encoding='utf-8').read()

VERSION = '0.2.0'

URL = 'http://github.com/QuantumElephant/NICE'

REQUIRES = ['numpy', 'scipy', 'cma']

PACKAGES = ['nice', 'nice.test']

CLASSIFIERS = [
    'Environment :: Console',
    'Intended Audience :: Science/Research',
    'Programming Language :: Python :: 3',
    'Topic :: Science/Engineering :: Molecular Science',
    ]

EXT_MODULES = [Extension(name='nice._kmc', sources=['nice/_kmc.f90'])]


if __name__ == '__main__':

    setup(
        name=NAME,
        license=LICENSE,
        author=AUTHOR,
        author_email=EMAIL,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        version=VERSION,
        url=URL,
        requires=REQUIRES,
        packages=PACKAGES,
        classifiers=CLASSIFIERS,
        ext_modules=EXT_MODULES,
        zip_safe=False,
        )
