# -*- coding: utf-8 -*-

import doctest
from datetime import date
import scippnexus
import os

from typing import Any, Dict, Optional
from docutils.nodes import document
from sphinx.application import Sphinx
import sphinx_book_theme


def add_buttons(
    app: Sphinx,
    pagename: str,
    templatename: str,
    context: Dict[str, Any],
    doctree: Optional[document],
):
    base = "https://scipp.github.io"
    l1 = []
    l1.append({"type": "link", "text": "scipp", "url": f"{base}"})
    l1.append({"type": "link", "text": "scippnexus", "url": f"{base}/scippnexus"})
    l1.append({"type": "link", "text": "scippneutron", "url": f"{base}/scippneutron"})
    l1.append({"type": "link", "text": "ess", "url": f"{base}/ess"})
    header_buttons = context["header_buttons"]
    header_buttons.append({
        "type": "group",
        "buttons": l1,
        "icon": "fa fa-caret-down",
        "text": "Related projects"
    })
    l2 = []
    l2.append({"type": "link", "text": "v0.1 (latest)", "url": f"{base}/scippnexus"})
    header_buttons.append({
        "type": "group",
        "buttons": l2,
        "icon": "fa fa-caret-down",
        "text": "Version"
    })


sphinx_book_theme.add_launch_buttons = add_buttons

html_show_sourcelink = True

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx_autodoc_typehints',
    'sphinx_copybutton',
    'nbsphinx',
]

autodoc_type_aliases = {
    'VariableLike': 'VariableLike',
    'MetaDataMap': 'MetaDataMap',
    'array_like': 'array_like',
}

rst_epilog = f"""
.. |SCIPP_RELEASE_MONTH| replace:: {date.today().strftime("%B %Y")}
.. |SCIPP_VERSION| replace:: {scippnexus.__version__}
"""  # noqa: E501

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipp': ('https://scipp.github.io/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'xarray': ('https://xarray.pydata.org/en/stable/', None)
}

# autodocs includes everything, even irrelevant API internals. autosummary
# looks more suitable in the long run when the API grows.
# For a nice example see how xarray handles its API documentation.
autosummary_generate = True

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = False
napoleon_preprocess_types = True
napoleon_type_aliases = {
    # objects without namespace: scipp
    "DataArray": "~scipp.DataArray",
    "Dataset": "~scipp.Dataset",
    "Variable": "~scipp.Variable",
    # objects without namespace: numpy
    "ndarray": "~numpy.ndarray",
}
typehints_defaults = 'comma'
typehints_use_rtype = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = '.rst'
html_sourcelink_suffix = ''  # Avoid .ipynb.txt extensions in sources

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = u'scippnexus'
copyright = u'2022 Scipp contributors'
author = u'Scipp contributors'

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = u''
# The full version, including alpha/beta/rc tags.
release = u''

warning_is_error = True

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = None

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False

# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#

html_theme = 'sphinx_book_theme'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
html_theme_options = {
    "logo_only": True,
    "repository_url": "https://github.com/scipp/scippnexus",
    "repository_branch": "main",
    "path_to_docs": "docs",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "show_toc_level": 2,  # Show subheadings in secondary sidebar
}

if 'OUTDATED_VERSION' in os.environ:
    html_theme_options["announcement"] = (
        "⚠️ You are viewing the documentation for an old version of scippnexus. "
        "Switch to <a href='https://github.com/scipp/scippnexus'>latest version.</a> ⚠️"
    )

html_logo = "_static/logo.png"
html_favicon = "_static/favicon.ico"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# -- Options for HTMLHelp output ------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'scippnexusdoc'

# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',

    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'scipp.tex', u'scipp Documentation', u'Simon Heybrock', 'manual'),
]

# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, 'scipp', u'scipp Documentation', [author], 1)]

# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'scipp', u'scipp Documentation', author, 'scipp',
     'One line description of project.', 'Miscellaneous'),
]

# -- Options for Matplotlib in notebooks ----------------------------------

nbsphinx_execute_arguments = [
    "--Session.metadata=scipp_docs_build=True",
]

# -- Options for doctest --------------------------------------------------

doctest_global_setup = '''
import numpy as np
import scipp as sc
'''

# Using normalize whitespace because many __str__ functions in scipp produce
# extraneous empty lines and it would look strange to include them in the docs.
doctest_default_flags = doctest.ELLIPSIS | doctest.IGNORE_EXCEPTION_DETAIL | \
                        doctest.DONT_ACCEPT_TRUE_FOR_1 | \
                        doctest.NORMALIZE_WHITESPACE

# -- Options for linkcheck ------------------------------------------------

linkcheck_ignore = [
    # Specific lines in Github blobs cannot be found by linkcheck.
    r'https?://github\.com/.*?/blob/[a-f0-9]+/.+?#',
]
