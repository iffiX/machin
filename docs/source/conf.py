# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join('..', '..')))

import machin

# -- Project information -----------------------------------------------------

project = 'Machin'
copyright = '2020, Iffi'
author = 'Iffi'

# The full version, including alpha/beta/rc tags
release = machin.__version__


# -- General configuration ---------------------------------------------------
master_doc = 'index'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.todo',
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosectionlabel',
]

# Add any paths that contain templates here, relative to this directory.
# templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# Both the class’ and the __init__ method’s docstring are concatenated
# and inserted.
autoclass_content = 'both'
autodoc_default_options = {
    #'special-members': '__call__, __getitem__, __len__'
}
autodoc_member_order = 'groupwise'  # 'bysource', 'alphabetical'
autodoc_typehints = "description"
autodoc_mock_imports = [""]
# autodoc_dumb_docstring = True

# same as autoclass_content = 'both',
# but __init__ signature is also documented, not beautiful.
# napoleon_include_init_with_doc = True
# napoleon_use_admonition_for_examples = True

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'theme'
html_logo = 'static/icon_title.png'
html_favicon = 'static/favicon.png'
html_theme_path = ['../']

html_theme_options = {
    'canonical_url': '',
    'logo_only': True,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    # Toc options
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['static']

# automatic section reference label
autosectionlabel_prefix_document = True
autosectionlabel_maxdepth = 3
numfig = True
