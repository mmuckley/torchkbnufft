# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
from datetime import date
import os
import sys

sys.path.insert(0, os.path.abspath("../.."))
os.environ["PYTORCH_JIT"] = "0"
year = date.today().year

import torchkbnufft


# -- Project information -----------------------------------------------------

project = "torchkbnufft"
copyright = f"{year}, Matthew Muckley"
author = "Matthew Muckley"

# The full version, including alpha/beta/rc tags
release = f"v{torchkbnufft.__version__}"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinxcontrib.katex",
    "sphinx.ext.autosectionlabel",
    "sphinx_autodoc_typehints",
]
# autodoc_mock_imports = ["torch"]

# build the templated autosummary files
autosummary_generate = True
numpydoc_show_class_members = False

# autosectionlabel throws warnings if section names are duplicated.
# The following tells autosectionlabel to not throw a warning for
# duplicated section names that are in different documents.
autosectionlabel_prefix_document = True

napoleon_use_ivar = True
# napoleon_use_param = True

napoleon_type_aliases = {
    "Tensor": ":py:class:`torch.Tensor`",
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

html_sidebars = {"**": ["globaltoc.html", "searchbox.html"]}

source_suffix = ".rst"

master_doc = "index"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

add_module_names = False

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": 1,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

coverage_ignore_functions = ["torch.jit"]


intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
}
