What is torch-pme
=================

``torch-pme`` provides an interface in which the ``positions`` of the atoms in a
structure are stored in :class:`torch.Tensor` objects, or in a :class:`metatensor.System
<metatensor.torch.atomistic.System>` object.

The primary goal is to design a library to compute long-range interactions that can be
easily integrated with existing short-range machine learning (ML) architectures,
essentially providing an easy-to-use framework to build range separated models for
atomistic machine learning. To this end, our reference ``torch-pme`` library provides

1. A modular implementation of range-separated potentials working for arbitrary unit
   cells including triclinic ones, as well as for systems with free (non-prediodic)
   boundary condistions.
2. Full integration with PyTorch, featuring differentiable and learnable parameters,
3. Efficient particle-mesh-based computation with automatic hyperparameter tuning,
4. Pure long-range descriptors, free of noisy short-range contributions,
5. Support for arbitrary invariant and equivariant features and ML architectures.

``torch-pme`` can be used as an *end-to-end* library computing the potential from
positions and charges and as a *modular* library to construct complex Fourier-domain
architectures. To use ``torch-pme`` as an end-to-end a library the main entry points are
:ref:`calculators`, that compute pair :ref:`potentials` combining real-space and k-space
components. They take a description of a structure as input and return the calculated
potential at the atomic positions as output. To use torch-pme as a modular library, we
provide a set of building blocks that can be combined to build custom range-separated
architectures, as shown in the figure below.

.. note::

   ``torch-pme`` adopts a different parameter convention. Instead of using the usual
   parameters, the inverse width of Gaussian charge cloud :math:`\alpha` and the Fourier
   space cutoff :math:`K`, we use the smearing :math:`\sigma=\frac{1}{\sqrt{2}\alpha}`
   and the long-range resolution :math:`h=\frac{2\pi}{K}`.

.. figure:: ../static/images/pme-structure.*
    :width: 650px
    :align: center

    A schematic representation of the main building blocks that are contained inside a
    :ref:`calculators` of a range-separated architecture, that combines an evaluation of
    the short-range part of the :ref:`potentials` :math:`v_\mathrm{SR}(r)` based on
    local interatomic distance information with the evaluation of the long-range part
    :math:`v_\mathrm{LR}(k)` using grids via a :ref:`mesh_interpolator` and a
    :ref:`kspace_filter`.
