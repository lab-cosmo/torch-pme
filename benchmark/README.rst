``benchmark``: performance sanity checking
==========================================

This provides a basic benchmarking script intended to catch major performance regressions (or improvements!). It emulates a "training"-style workload of computing energy and forces for a number of similar systems with pre-computed neighborlists and settings. In particular, we run PME forward and backward calculations for supercells of cubic CsCl crystals, in different sizes. Results are stored as a time-stamped ``.yaml`` file, together with some basic system and version information (which is just the output of ``git`` for now).


Usage
-----

First, make sure to install ``torch-pme`` with the right dependencies in a fresh enviroment:

.. code-block:: bash

    pip install .[metatensor,examples]

Then, run

.. code-block:: bash

    python run.py


Please include the output ``.yaml`` file in any major pull request. We recommend using a H100 as benchmark GPU (internally, we use the ``kuma`` cluster).
