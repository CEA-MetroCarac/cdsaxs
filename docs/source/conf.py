# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
sys.path.insert(0, os.path.abspath('../../'))  # Adjust the path as needed

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
# Add any Sphinx extension module names here, as strings.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',  # To support Google and NumPy style docstrings
    'sphinx_rtd_theme',
]

# Generate autosummary pages automatically
autosummary_generate = True

# List of patterns, relative to source directory, that match files and directories to ignore.
exclude_patterns = []

# The suffix of source filenames.
source_suffix = ['.rst']

# The master toctree document.
master_doc = 'index'

project = 'cdsaxs'
copyright = '2024, Nischal Dhungana'
author = 'Nischal Dhungana'
release = 'v1.0.0'


templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_theme_options = {
    # Toc options
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False,
}