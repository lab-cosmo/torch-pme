# Pseudo-sphinx configuration to run sphinx-gallery as a command line tool

import os
from sphinx_gallery.sorting import FileNameSortKey
from chemiscope.sphinx import ChemiscopeScraper

extensions = [
    "sphinx_gallery.gen_gallery",
    "chemiscope.sphinx",
]

HERE = os.path.dirname(__file__)
ROOT = os.path.realpath(os.path.join(HERE, "..", ".."))

sphinx_gallery_conf = {
    "examples_dirs": ["../../examples"],
    "filename_pattern": ".*",
    "gallery_dirs": ["examples"],
    "image_scrapers": ["matplotlib", ChemiscopeScraper()],
    "prefer_full_module": ["torchpme"],
    "reference_url": {"torchpme": None},
    "remove_config_comments": True,
    "within_subsection_order": FileNameSortKey,
}
