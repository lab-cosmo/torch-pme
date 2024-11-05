import os
import subprocess
import sys
from datetime import datetime

import tomli

# When importing metatensor-torch, this will change the definition of the classes
# to include the documentation
os.environ["METATENSOR_IMPORT_FOR_SPHINX"] = "1"

import torchpme  # noqa

suppress_warnings = ["config.cache"]

ROOT = os.path.abspath(os.path.join("..", ".."))

# We use a second (pseudo) sphinx project located in `docs/generate_examples` to run the
# examples and generate the actual output for our shinx-gallery. This is necessary
# because here we have to set `METATENSOR_IMPORT_FOR_SPHINX` to `"1"` allowing the
# correct generation of the class and function docstrings which are seperate from the
# actual code.
#
# We register and use the same sphinx gallery configuration as in the pseudo project.
sys.path.append(os.path.join(ROOT, "docs"))
from generate_examples.conf import sphinx_gallery_conf  # noqa


# -- Project information -----------------------------------------------------

# The master toctree document.
master_doc = "index"

with open(os.path.join(ROOT, "pyproject.toml"), "rb") as fp:
    project_dict = tomli.load(fp)["project"]

project = project_dict["name"]
author = ", ".join(a["name"] for a in project_dict["authors"])

copyright = f"{datetime.now().date().year}, {author}"

# The full version, including alpha/beta/rc tags
release = torchpme.__version__


# -- General configuration ---------------------------------------------------


def generate_examples():
    # We can not run sphinx-gallery in the same process as the normal sphinx, since they
    # need to import metatensor.torch differently (with and without
    # METATENSOR_IMPORT_FOR_SPHINX=1). So instead we run it inside a small script, and
    # include the corresponding output later.
    del os.environ["METATENSOR_IMPORT_FOR_SPHINX"]
    script = os.path.join(ROOT, "docs", "generate_examples", "generate-examples.py")
    subprocess.run([sys.executable, script])
    os.environ["METATENSOR_IMPORT_FOR_SPHINX"] = "1"


def setup(app):
    generate_examples()


# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_gallery.gen_gallery",
    "sphinx_toggleprompt",
    "chemiscope.sphinx",
]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "Thumbs.db",
    ".DS_Store",
    "examples/sg_execution_times.rst",
    "examples/ase/index.rst",
    "sg_execution_times.rst",
]


python_use_unqualified_type_names = True

autoclass_content = "class"
autodoc_member_order = "bysource"
autodoc_typehints = "both"
autodoc_typehints_format = "short"

intersphinx_mapping = {
    "ase": ("https://wiki.fysik.dtu.dk/ase/", None),
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "metatensor": ("https://docs.metatensor.org/latest/", None),
    "vesin": ("https://luthaf.fr/vesin/latest/", None),
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

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["../static"]
html_favicon = "../static/torch-pme-icon.png"
html_logo = "../static/torch-pme-icon.svg"

# font-awesome logos (used in the footer)
html_css_files = [
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/fontawesome.min.css",
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/solid.min.css",
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/brands.min.css",
]
