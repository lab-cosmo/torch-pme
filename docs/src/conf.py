import os
import sys
from datetime import datetime

import tomli  # Replace by tomllib from std library once docs are build with Python 3.11

import meshlode


ROOT = os.path.abspath(os.path.join("..", ".."))
sys.path.insert(0, ROOT)

# -- Project information -----------------------------------------------------

# The master toctree document.
master_doc = "index"

with open(os.path.join(ROOT, "pyproject.toml"), "rb") as fp:
    project_dict = tomli.load(fp)["project"]

project = project_dict["name"]
author = ", ".join(a["name"] for a in project_dict["authors"])

copyright = f"{datetime.now().date().year}, {author}"

# The full version, including alpha/beta/rc tags
release = meshlode.__version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_gallery.gen_gallery",
    "sphinx_toggleprompt",
]

sphinx_gallery_conf = {
    "filename_pattern": "/*",
    "examples_dirs": ["../../examples"],
    "gallery_dirs": ["examples"],
    "min_reported_time": 60,
    "reference_url": {"meshlode": None},
    "prefer_full_module": ["meshlode"],
}

autoclass_content = "both"
autodoc_member_order = "bysource"
autodoc_typehints = "both"
autodoc_typehints_format = "short"

intersphinx_mapping = {
    "ase": ("https://wiki.fysik.dtu.dk/ase/", None),
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "metatensor": ("https://lab-cosmo.github.io/metatensor/latest/", None),
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"

html_theme_options = {
    "footer_icons": [
        {
            "name": "GitHub",
            "url": project_dict["urls"]["repository"],
            "html": "",
            "class": "fa-brands fa-github fa-2x",
        },
    ],
}

# font-awesome logos (used in the footer)
html_css_files = [
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/fontawesome.min.css",
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/solid.min.css",
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/brands.min.css",
]
