# To preview the docs locally
# cd docs
# sphinx-build -b html . _build/html
# or 
# sphinx-build -b html -a . _build/html

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "axionbloch"
author = "Yuzhe Zhang"
release = "0.1.0"


extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "myst_parser",
]

source_suffix = {
    ".md": "markdown",
}

master_doc = 'index'

# # Enable MyST extensions
# myst_enable_extensions = [
#     "amsmath",         # allows $$ ... $$ blocks
# ]
# myst_dmath_double_inline = True  # also supports $$...$$

html_theme = "sphinx_rtd_theme"