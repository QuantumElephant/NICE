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

import sphinx_rtd_theme

import nice


project = 'NICE'


copyright = '2020, Ayers Lab'


author = 'Ayers Lab'


release = '0.2.0'


extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    ]


templates_path = [
    '_templates',
    ]


exclude_patterns = [
    ]


html_theme = 'sphinx_rtd_theme'


html_static_path = [
    '_static',
    ]


mathjax_path = 'https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.6/MathJax.js'


mathjax_config = {
    'extensions': ['tex2jax.js'],
    'jax': ['input/TeX', 'output/HTML-CSS'],
    }
