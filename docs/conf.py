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
from distutils.version import LooseVersion

import sphinx
from m2r import MdInclude

sys.path.insert(0, os.path.abspath(os.path.pardir))
sys.path.insert(0, os.path.abspath("sphinxext"))

from github_link import make_linkcode_resolve

import idconn

# from m2r import MdInclude

sys.path.insert(0, os.path.abspath(os.path.pardir))
sys.path.insert(0, os.path.abspath("sphinxext"))

# import idconn

# -- Project information -----------------------------------------------------

project = "IDConn"
copyright = "2020, Katherine Bottenhorn"
author = "Katherine Bottenhorn"

# The full version, including alpha/beta/rc tags
version = idconn.__version__
# The full version, including alpha/beta/rc tags.
release = idconn.__version__

# -- General configuration ---------------------------------------------------

autosummary_generate = True
add_module_names = False

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",  # standard
    "sphinx.ext.autosummary",  # standard
    "sphinx.ext.intersphinx",  # links code to other packages
    "sphinx.ext.linkcode",  # links to code from api
    "sphinx.ext.napoleon",  # alternative to numpydoc
    "sphinx_copybutton",  # for copying code snippets
    "sphinx_gallery.gen_gallery",  # example gallery
    "sphinxarg.ext",  # argparse
    "sphinxcontrib.bibtex",  # for foot-citations
    "recommonmark",  # markdown parser
]

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
# Napoleon settings
# -----------------------------------------------------------------------------
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_keyword = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# -----------------------------------------------------------------------------
# intersphinx
# -----------------------------------------------------------------------------
_python_version_str = "{0.major}.{0.minor}".format(sys.version_info)
_python_doc_base = "https://docs.python.org/" + _python_version_str
intersphinx_mapping = {
    "python": (_python_doc_base, None),
    "numpy": ("https://numpy.org/doc/stable/", (None, "./_intersphinx/numpy-objects.inv")),
    "scipy": (
        "https://docs.scipy.org/doc/scipy/reference",
        (None, "./_intersphinx/scipy-objects.inv"),
    ),
    "sklearn": ("https://scikit-learn.org/stable", (None, "./_intersphinx/sklearn-objects.inv")),
    "matplotlib": ("https://matplotlib.org/", (None, "https://matplotlib.org/objects.inv")),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "nibabel": ("https://nipy.org/nibabel/", None),
    "nilearn": ("http://nilearn.github.io/stable/", None),
}

# -----------------------------------------------------------------------------
# HTMLHelp output
# -----------------------------------------------------------------------------
# Output file base name for HTML help builder.
htmlhelp_basename = "idconndoc"

# The following is used by sphinx.ext.linkcode to provide links to github
linkcode_resolve = make_linkcode_resolve(
    "idconn",
    "https://github.com/62442katieb/IDConn/blob/{revision}/{package}/{path}#L{lineno}",
)

# -----------------------------------------------------------------------------
# Sphinx gallery
# -----------------------------------------------------------------------------
sphinx_gallery_conf = {
    # path to your examples scripts
    "examples_dirs": "../examples",
    # run examples with a number, then "plot"
    "filename_pattern": "/[0-9]+_plot_",
    # path where to save gallery generated examples
    "gallery_dirs": "auto_examples",
    "backreferences_dir": "generated",
    # Modules for which function level galleries are created.
    # In this case sphinx_gallery and numpy in a tuple of strings.
    "doc_module": ("idconn"),
    "ignore_pattern": r"utils/.",
    "reference_url": {
        # The module you locally document uses None
        "idconn": None
    },
    "remove_config_comments": True,
}

# Generate the plots for the gallery
plot_gallery = "True"

# -----------------------------------------------------------------------------
# sphinxcontrib-bibtex
# -----------------------------------------------------------------------------
bibtex_bibfiles = ["./references.bib"]
bibtex_style = "unsrt"
bibtex_reference_style = "author_year"
bibtex_footbibliography_header = ""


def setup(app):
    """From https://github.com/rtfd/sphinx_rtd_theme/issues/117"""
    app.add_css_file("theme_overrides.css")
    app.add_css_file("nimare.css")
    app.connect("autodoc-process-docstring", generate_example_rst)
    # Fix to https://github.com/sphinx-doc/sphinx/issues/7420
    # from https://github.com/life4/deal/commit/7f33cbc595ed31519cefdfaaf6f415dada5acd94
    # from m2r to make `mdinclude` work
    app.add_config_value("no_underscore_emphasis", False, "env")
    app.add_config_value("m2r_parse_relative_links", False, "env")
    app.add_config_value("m2r_anonymous_references", False, "env")
    app.add_config_value("m2r_disable_inline_math", False, "env")
    app.add_directive("mdinclude", MdInclude)


def generate_example_rst(app, what, name, obj, options, lines):
    # generate empty examples files, so that we don't get
    # inclusion errors if there are no examples for a class / module
    folder = os.path.join(app.srcdir, "generated")
    if not os.path.isdir(folder):
        os.makedirs(folder)
    examples_path = os.path.join(app.srcdir, "generated", "%s.examples" % name)
    if not os.path.exists(examples_path):
        # touch file
        open(examples_path, "w").close()
