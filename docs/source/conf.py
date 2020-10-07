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
import os
import sys

from datetime import datetime
from distutils.version import LooseVersion

import sphinx

# from m2r import MdInclude

sys.path.insert(0, os.path.abspath(os.path.pardir))
sys.path.insert(0, os.path.abspath("sphinxext"))

# import idconn

# -- Project information -----------------------------------------------------

project = "IDConn"
copyright = "2020, Katherine Bottenhorn"
author = "Katherine Bottenhorn"

# The full version, including alpha/beta/rc tags
release = "0.3dev"


# -- General configuration ---------------------------------------------------

autosummary_generate = True
add_module_names = False

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ["sphinx.ext.autodoc"]

if LooseVersion(sphinx.__version__) < LooseVersion("1.4"):
    extensions.append("sphinx.ext.pngmath")
else:
    extensions.append("sphinx.ext.imgmath")

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

source_suffix = ".rst"
master_doc = "index"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", ".DS_Store", "utils/*"]

# The reST default role (used for this markup: `text`) to use for all documents.
# default_role = "autolink"

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "alabaster"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# html_favicon = "_static/logo-transparent.png"
html_logo = "_static/logo-transparent.png"

# -----------------------------------------------------------------------------
# intersphinx
# -----------------------------------------------------------------------------
_python_version_str = "{0.major}.{0.minor}".format(sys.version_info)
_python_doc_base = "https://docs.python.org/" + _python_version_str
intersphinx_mapping = {
    "python": (_python_doc_base, None),
    "numpy": (
        "https://docs.scipy.org/doc/numpy",
        (None, "./_intersphinx/numpy-objects.inv"),
    ),
    "scipy": (
        "https://docs.scipy.org/doc/scipy/reference",
        (None, "./_intersphinx/scipy-objects.inv"),
    ),
    "sklearn": (
        "https://scikit-learn.org/stable",
        (None, "./_intersphinx/sklearn-objects.inv"),
    ),
    "matplotlib": (
        "https://matplotlib.org/",
        (None, "https://matplotlib.org/objects.inv"),
    ),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "nibabel": ("https://nipy.org/nibabel/", None),
    "nilearn": ("http://nilearn.github.io/", None),
}

# -----------------------------------------------------------------------------
# Sphinx gallery
# -----------------------------------------------------------------------------
sphinx_gallery_conf = {
    # path to your examples scripts
    "examples_dirs": "../examples",
    # path where to save gallery generated examples
    "gallery_dirs": "auto_examples",
    "backreferences_dir": "generated",
    # Modules for which function level galleries are created.  In
    # this case sphinx_gallery and numpy in a tuple of strings.
    "doc_module": ("nimare"),
    "ignore_patterns": ["utils/"],
    "reference_url": {
        # The module you locally document uses None
        "nimare": None,
    },
}

# Generate the plots for the gallery
plot_gallery = "True"
