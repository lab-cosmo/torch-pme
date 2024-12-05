torch-pme
=========

.. image:: https://raw.githubusercontent.com/lab-cosmo/torch-pme/refs/heads/main/docs/src/logo/torch-pme.svg
     :width: 200 px
     :align: left

|tests| |codecov| |docs|

.. marker-introduction

``torch-pme`` enables efficient, auto-differentiable computation of long-range
interactions in PyTorch. Auto-differentiation is supported for particle *positions*,
*charges*, and *cell* parameters, allowing not only the automatic computation of forces
but also enabling general applications in machine learning tasks. The library offers
classes for Particle-Particle Particle-Mesh Ewald (P3M), Particle Mesh Ewald (``PME``),
standard ``Ewald``, and non-periodic methods, with the flexibility to calculate
potentials beyond :math:`1/r` electrostatics, including arbitrary order :math:`1/r^p`
potentials.

Optimized for both CPU and GPU devices, ``torch-pme`` is fully `TorchScriptable`_,
allowing it to be converted into a format that runs independently of Python, such as in
C++, making it ideal for high-performance production environments.

.. _`TorchScriptable`: https://pytorch.org/docs/stable/jit.html

.. marker-documentation

For details, tutorials, and examples, please have a look at our `documentation`_.

.. _`documentation`: https://lab-cosmo.github.io/torch-pme

.. marker-installation

Installation
------------

You can install *torch-pme* using pip with

.. code-block:: bash

    pip install torch-pme

and ``import torchpme`` to use it in your projects!

We also provide bindings to `metatensor <https://docs.metatensor.org>`_ which
can optionally be installed together and used as ``torchpme.metatensor`` via

.. code-block:: bash

    pip install torch-pme[metatensor]

.. marker-issues

Having problems or ideas?
-------------------------

Having a problem with torch-pme? Please let us know by `submitting an issue
<https://github.com/lab-cosmo/torch-pme/issues>`_.

Submit new features or bug fixes through a `pull request
<https://github.com/lab-cosmo/torch-pme/pulls>`_.

.. marker-cite

Reference
---------

If you use *torch-pme* for your work, please read and cite our preprint available on
`arXiv`_.

.. code-block::

   @article{loche_fast_2024,
      title = {Fast and Flexible Range-Separated Models for Atomistic Machine Learning},
      author = {Loche, Philip and {Huguenin-Dumittan}, Kevin K. and Honarmand, Melika and Xu, Qianjun and Rumiantsev, Egor and How, Wei Bin and Langer, Marcel F. and Ceriotti, Michele},
      year = {2024},
      month = dec,
      number = {arXiv:2412.03281},
      eprint = {2412.03281},
      primaryclass = {physics},
      publisher = {arXiv},
      doi = {10.48550/arXiv.2412.03281},
      urldate = {2024-12-05},
      archiveprefix = {arXiv}
      }

.. _`arXiv`: http://arxiv.org/abs/2412.03281

.. marker-contributing

Contributors
------------

Thanks goes to all people that make torch-pme possible:

.. image:: https://contrib.rocks/image?repo=lab-cosmo/torch-pme
   :target: https://github.com/lab-cosmo/torch-pme/graphs/contributors

.. |tests| image:: https://github.com/lab-cosmo/torch-pme/workflows/Tests/badge.svg
   :alt: Github Actions Tests Job Status
   :target: https://github.com/lab-cosmo/torch-pme/actions?query=workflow%3ATests

.. |codecov| image:: https://codecov.io/gh/lab-cosmo/torch-pme/graph/badge.svg?token=srVKRy7r6m
   :alt: Code coverage
   :target: https://codecov.io/gh/lab-cosmo/torch-pme

.. |docs| image:: https://img.shields.io/badge/documentation-latest-sucess
   :alt: Documentation
   :target: `documentation`_
