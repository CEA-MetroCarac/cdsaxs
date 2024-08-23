# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys

sys.path.insert(0, os.path.abspath('../../'))  # Adjust the path as needed

html_extra_path = [
    os.path.abspath('../../assets/images'),os.path.abspath('../../Tutorials')   # Absolute path to your images directory
]

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
# Add any Sphinx extension module names here, as strings.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',  # To support Google and NumPy style docstrings
    'sphinx_rtd_theme',
    'sphinx.ext.mathjax',
    'nbsphinx',
    'sphinx.ext.viewcode',
    'sphinx.ext.todo',
]

#nbshpinx configuration
nbsphinx_allow_errors = True  # Continue rendering even if there are errors in the notebooks
nbsphinx_execute = 'never'    # Don't execute the notebooks on build; use the pre-executed output


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
release = 'v0.0.1'


templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'

html_theme_options = {
    # Toc options
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False,
}


html_static_path = ['_static']
