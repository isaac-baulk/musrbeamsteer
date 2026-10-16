# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import musrbeamsteer
from datetime import date

project = 'musrbeamsteer'
copyright = f'{date.today().year}, {musrbeamsteer.__author__}'
author = musrbeamsteer.__author__
release = musrbeamsteer.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.apidoc",
    "myst_parser",
]

templates_path = ['_templates']
exclude_patterns = []

apidoc_modules = [
    {"path": "../../musrbeamsteer",
     "destination": "api",}
]

autodoc_typehints = 'both'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
