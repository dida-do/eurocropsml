import re
from importlib import metadata

# -- General configuration ------------------------------------------------

# In nitpick mode (-n), still ignore any of the following "broken" references
# to non-types.
nitpick_ignore = [
    ("py:class", "Any value"),
    ("py:class", "callable"),
    ("py:class", "callables"),
    ("py:class", "tuple of types"),

    # some external references could not be resolved, we ignore them
    
    # pydantic references
    ("py:class", "ComputedFieldInfo"),
    ("py:class", "ConfigDict"),
    ("py:class", "FieldInfo"),
]

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "myst_parser",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_tippy",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinxcontrib.mermaid",
]

# Generate referenced autodoc stub files for autosummary entries
autosummary_generate = True

# Exclude CLI modules from autosummary, they will be documented via typer-cli
autosummary_mock_imports = [
    "eurocropsml.cli",
    "eurocropsml.acquisition.cli",
    "eurocropsml.dataset.cli",
]

# Add any MyST extension names here, as strings.
myst_enable_extensions = [
    "colon_fence",
    "smartquotes",
    "deflist",
    "attrs_inline",
    "dollarmath",
    "amsmath", 
]

# Exclude code and console prompts from copybuttons
copybutton_exclude = '.linenos, .gp'

# Configure sphinx-tippy tooltips
tippy_skip_anchor_classes = ("headerlink", "sd-stretched-link", "sd-rounded-pill")
tippy_anchor_parent_selector = "div.content"  # for Furo theme

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The master toctree document.
master_doc = "index"

# General information about the project.
project = "eurocropsml"
author = "Joana Reuss, Jan Macdonald"
copyright = f"2024, {author}"

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.

# The full version, including alpha/beta/rc tags.
release = metadata.version("eurocropsml")
if "dev" in release:
    release = version = "UNRELEASED"
else:
    # The short X.Y version.
    version = release.rsplit(".", 1)[0]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ["_build", "_templates"]

# The reST default role (used for this markup: `text`) to use for all
# documents.
default_role = None

# If true, '()' will be appended to :func: etc. cross-reference text.
add_function_parentheses = True

# If true 'package.module.Class' renders as 'Class' etc.
add_module_names = False

# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

html_theme = "furo"
html_theme_options = {
    "sidebar_hide_name": False,
    "top_of_page_buttons": [],
}
html_css_files = [
    "custom.css",
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.1.1/css/all.min.css",
]

# Customize the sidebar title (default is <project> <version> documentation)
html_title = "EuroCropsML Documentation"

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
# html_favicon = None

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# If false, no module index is generated.
html_domain_indices = True

# If false, no index is generated.
html_use_index = True

# If true, the index is split into individual pages for each letter.
html_split_index = False

# If true, links to the reST sources are added to the pages.
html_show_sourcelink = False

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
html_show_sphinx = True

# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
html_show_copyright = True

# If true, an OpenSearch description file will be output, and all pages will
# contain a <link> tag referring to it.  The value of this option must be the
# base URL from which the finished HTML is served.
# html_use_opensearch = ''

# Output file base name for HTML help builder.
htmlhelp_basename = "eurocropsml-doc"

# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [("index", "eurocropsml", "EuroCropsML Documentation", ["Joana Reuss", "Jan Macdonald"], 1)]


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        "index",
        "eurocropsml",
        "EuroCropsML Documentation",
        "Joana Reuss, Jan Macdonald",
        "eurocropsml",
        "Preprocessed EuroCrops dataset to benchmark few-shot crop type classification.",
        "Miscellaneous",
    )
]

epub_description = "Preprocessed EuroCrops dataset to benchmark few-shot crop type classification."

intersphinx_mapping = {
    "geopandas": ("https://geopandas.org/en/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "pydantic": ("https://docs.pydantic.dev/latest/", None),
    "python": ("https://docs.python.org/3", None),
    "requests": ("https://requests.readthedocs.io/en/latest/", None),
    "shapely": ("https://shapely.readthedocs.io/en/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "mmap_ninja": ("https://mmapninja.readthedocs.io/en/latest/", None),
}

# Allow non-local URIs so we can have images in CHANGELOG etc.
suppress_warnings = ["image.nonlocal_uri"]


# -- Custom autodoc event handlers -------------------------------------------

def _custom_process_docstring(app, what, name, obj, options, lines) -> None:
    """" Can overwrite current docstring lines by modifying them in-place. """
   
    # pydantic specific overwrites
    if any(["pydantic" in line for line in lines]):
        # pydantic uses `mkdoc`` documentation and `mkdocstrings`` for parsing
        # docstrings. They use a non-standard [reference][target] syntax instead
        # of the common markdown [reference](target) syntax, thus not correctly 
        # handled by the `myst_parser`. We need to overwrite them as valid 
        # reStructured text references at this point of the parsing
        match_pattern = r"\[(?P<reference>[^]]+)\]\[(?P<target>[^]]+)\]"
        def _subsitution(g) -> str:
            # remove special characters from the reference
            cleaned_reference = re.sub(r"[\W]", "", g.group("reference"))
            # targets should already by valid paths, nothing to clean here
            cleaned_target = g.group("target")
            return f":any:`{cleaned_reference}<{cleaned_target}>`"

        for idx, line in enumerate(lines):
            new_line = re.sub(match_pattern, _subsitution, line)
            if new_line != line:
                lines[idx] = new_line

# connect handlers to Sphinx app
def setup(app):
    app.connect("autodoc-process-docstring", _custom_process_docstring)