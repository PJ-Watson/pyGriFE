"""
Configuration file for generating documentation with sphinx.
"""

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "PyGriFE"
copyright = "2025, Peter J. Watson"
author = "Peter J. Watson"
# release = '0.0.3'
from importlib.metadata import version as get_version

release: str = get_version("pygrife")
# for example take major/minor
version: str = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join("..", "..", "src")))

os.environ["GRIZLI"] = ""
os.environ["iref"] = ""
os.environ["jref"] = ""

try:
    from sphinx_astropy.conf.v2 import *
except ImportError:
    print(
        "ERROR: the documentation requires the sphinx-astropy package to be installed."
    )
    sys.exit(1)

extensions = [
    "sphinx.ext.autosummary",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.intersphinx",
    "numpydoc",
    "sphinx_copybutton",
]

# Don't show typehints in description or signature
autodoc_typehints = "none"
autodoc_member_order = "bysource"

# Create xrefs for parameter types in docstrings
numpydoc_xref_param_type = True
# Don't make a table of contents for all class methods and attributes
numpydoc_class_members_toctree = False
# Don't show all class members in the methods and attributes list
numpydoc_show_class_members = False
# Don't show inherited class members either
numpydoc_show_inherited_class_members = False
# autosummary_generate=True

version_link = f"{sys.version_info.major}.{sys.version_info.minor}"
intersphinx_mapping = {
    "python": (
        f"https://docs.python.org/{version_link}",
        None,
    ),  # link to used Python version
    "grizli": ("https://grizli.readthedocs.io/en/latest/", None),
    "PyQt6": ("https://www.riverbankcomputing.com/static/Docs/PyQt6/", None),
}

# Avoid ambiguous section headings, prefix with document name
autosectionlabel_prefix_document = True

templates_path = ["_templates"]
exclude_patterns = []

# Any `...` defaults to a link
default_role = "autolink"


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
