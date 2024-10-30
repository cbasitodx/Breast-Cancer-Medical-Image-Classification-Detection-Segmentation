# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0, os.path.abspath("../src"))
sys.path.insert(1, os.path.abspath("../src/dir_manager"))
sys.path.insert(2, os.path.abspath("../src/models"))
sys.path.insert(3, os.path.abspath("../src/viewer"))

project = 'Breast Cancer Medical Image Classification, Detection & Segmentation'
copyright = '2024, Sebastián Kay Conde Lorenzo, Jaime Capdepon Fraile, Joaquín Negrete Saab, Yang Liu, Christian Most Tazon'
author = 'Sebastián Kay Conde Lorenzo, Jaime Capdepon Fraile, Joaquín Negrete Saab, Yang Liu, Christian Most Tazon'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
