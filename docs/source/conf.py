# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import simba_ml
import os
import sys
sys.path.append(os.path.abspath('../..'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'SimbaML'
copyright = '2022, Maximilian Kleissl; Björn Heyder; Julian Zabbarov; Lukas Drews'

author = 'Maximilian Kleissl, Björn Heyder, Julian Zabbarov, Lukas Drews'
release = simba_ml.__version__
version = simba_ml.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.napoleon',
    'sphinx.ext.duration',
    'sphinx.ext.autodoc',
    'sphinx_copybutton',
    'sphinx_design',
    'sphinx.ext.mathjax',
    'sphinx.ext.autosummary',
    'sphinx_design',
    'sphinx.ext.doctest'
]

autosummary_generate = True
autoclass_content = "both"
autodoc_inherit_docstrings = True

autodoc_default_options = {
    'members': True,
    'show-inheritance': True,
    'inherited-members': True,
    'no-special-members': True,
}

templates_path = ['_templates']
exclude_patterns = []
add_module_names = False

html_theme = 'furo'
#html_theme = "pydata_sphinx_theme"
html_logo = "_static/simba_ml_logo.jpg"
html_favicon = "_static/simba_ml_logo.jpg"
html_static_path = ['_static']

html_theme_options = {
    "source_edit_link": "https://github.com/DILiS-lab/SimbaML/-/blob/main/docs/source/{filename}",
}

master_doc = "contents"

nbsphinx_execute = "always"
