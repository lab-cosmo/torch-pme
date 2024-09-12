# Pseudo-sphinx configuration to run sphinx-gallery as a command line tool

import os
import chemiscope
from chemiscope.sphinx import ChemiscopeScraper

extensions = [
    "sphinx_gallery.gen_gallery",
    "chemiscope.sphinx",
]

HERE = os.path.dirname(__file__)
ROOT = os.path.realpath(os.path.join(HERE, "..", ".."))

sphinx_gallery_conf = {
    "filename_pattern": "/*",
    "examples_dirs": ["../../examples"],
    "gallery_dirs": ["examples"],
    "min_reported_time": 60,
    "reference_url": {"torchpme": None},
    "prefer_full_module": ["torchpme"],
    "image_scrapers" :("matplotlib", ChemiscopeScraper()),    
}
