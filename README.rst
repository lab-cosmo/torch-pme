torch-pme
=========

.. image:: https://raw.githubusercontent.com/lab-cosmo/torch-pme/refs/heads/main/docs/logo/torch-pme.svg
   :width: 200 px
   :align: left

|tests| |codecov| |docs|

.. marker-introduction

``torch-pme`` enables efficient and auto-differentiable computation of long-range
interactions in *PyTorch*. Auto-differentiation is supported for particle *positions*,

*charges*/*dipoles*, and *cell* parameters, allowing not only the automatic computation
of forces but also enabling general applications in machine learning tasks. For
**monopoles** the library offers classes for Particle-Particle Particle-Mesh Ewald
(``P3M``), Particle Mesh Ewald (``PME``), standard ``Ewald``, and non-periodic methods.
The library has the flexibility to calculate potentials beyond :math:`1/r`
electrostatics, including arbitrary order :math:`1/r^p` potentials. For **dipolar**
interaction we offer to calculate the :math:`1/r^3` potential using the standard
``Ewald`` method.

Optimized for both CPU and GPU devices, ``torch-pme`` is fully `TorchScriptable`_,
allowing it to be converted into a format that runs independently of Python, such as in
C++, making it ideal for high-performance production environments.

We also provide an experimental implementation for *JAX* in `jax-pme`_.

.. _`TorchScriptable`: https://pytorch.org/docs/stable/jit.html
.. _`jax-pme`: https://github.com/lab-cosmo/jax-pme

.. marker-documentation

Documentation
-------------

For details, tutorials, and examples, please have a look at our `documentation`_.

.. _`documentation`: https://lab-cosmo.github.io/torch-pme

.. marker-installation

Installation
------------

You can install *torch-pme* using pip with

.. code-block:: bash

    pip install torch-pme

or conda

.. code-block:: bash

    conda install -c conda-forge torch-pme

and ``import torchpme`` to use it in your projects!

We also provide bindings to `metatensor <https://docs.metatensor.org>`_ which can
optionally be installed together and used as ``torchpme.metatensor`` via

.. code-block:: bash

    pip install torch-pme[metatensor]

.. marker-quickstart

Quickstart
----------

Here is a simple example to get started with *torch-pme*:

.. code-block:: python

   >>> import torch
   >>> import torchpme

   >>> # Single charge in a cubic box
   >>> positions = torch.zeros((1, 3))
   >>> cell = 8 * torch.eye(3)
   >>> charges = torch.tensor([[1.0]])

   >>> # No neighbors for a single atom; use `vesin` for neighbors if needed
   >>> neighbor_indices = torch.zeros((0, 2), dtype=torch.int64)
   >>> neighbor_distances = torch.zeros((0,))

   >>> # Tune P3M parameters
   >>> smearing, p3m_parameters, _ = torchpme.tuning.tune_p3m(
   ...    charges=charges,
   ...    cell=cell,
   ...    positions=positions,
   ...    cutoff=5.0,
   ...    neighbor_indices=neighbor_indices,
   ...    neighbor_distances=neighbor_distances,
   ... )

   >>> # Initialize potential and calculator
   >>> potential = torchpme.CoulombPotential(smearing)
   >>> calculator = torchpme.P3MCalculator(potential, **p3m_parameters)

   >>> # Start recording operations done to ``positions``
   >>> _ = positions.requires_grad_()

   >>> # Compute (per-atom) potentials
   >>> potentials = calculator.forward(
   ...    charges=charges,
   ...    cell=cell,
   ...    positions=positions,
   ...    neighbor_indices=neighbor_indices,
   ...    neighbor_distances=neighbor_distances,
   ... )

   >>> # Calculate total energy and forces
   >>> energy = torch.sum(charges * potentials)
   >>> energy.backward()
   >>> forces = -positions.grad

For more examples and details, please refer to the `documentation`_.

.. marker-issues

Having problems or ideas?
-------------------------

Having a problem with *torch-pme*? Please let us know by `submitting an issue
<https://github.com/lab-cosmo/torch-pme/issues>`_.

Submit new features or bug fixes through a `pull request
<https://github.com/lab-cosmo/torch-pme/pulls>`_.

.. marker-cite

Reference
---------

If you use *torch-pme* for your work, please read and cite our publication available on
`JCP`_.

.. code-block::

   @article{10.1063/5.0251713,
      title = {Fast and flexible long-range models for atomistic machine learning},
      author = {Loche, Philip and Huguenin-Dumittan, Kevin K. and Honarmand, Melika and Xu, Qianjun and Rumiantsev, Egor and How, Wei Bin and Langer, Marcel F. and Ceriotti, Michele},
      journal = {The Journal of Chemical Physics},
      volume = {162},
      number = {14},
      pages = {142501},
      year = {2025},
      month = {04},
      issn = {0021-9606},
      doi = {10.1063/5.0251713},
      url = {https://doi.org/10.1063/5.0251713},
   }

.. _`JCP`: https://doi.org/10.1063/5.0251713

.. marker-contributing

Contributors
------------

Thanks goes to all people that make *torch-pme* possible:

.. image:: https://contrib.rocks/image?repo=lab-cosmo/torch-pme
   :target: https://github.com/lab-cosmo/torch-pme/graphs/contributors

.. |tests| image:: https://github.com/lab-cosmo/torch-pme/workflows/Tests/badge.svg
   :alt: Github Actions Tests Job Status
   :target: https://github.com/lab-cosmo/torch-pme/actions?query=branch%3Amain

.. |codecov| image:: https://codecov.io/gh/lab-cosmo/torch-pme/graph/badge.svg?token=srVKRy7r6m
   :alt: Code coverage
   :target: https://codecov.io/gh/lab-cosmo/torch-pme

.. |docs| image:: https://img.shields.io/badge/documentation-latest-sucess
   :alt: Documentation
   :target: `documentation`_
