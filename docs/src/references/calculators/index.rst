.. _calculators:

Calculators
###########

Below is a list of all calculators available. Calculators are the core of MeshLODE and
are algorithms for transforming Cartesian coordinates into representations suitable for
machine learning.

Our calculator API follows the `rascaline <https://luthaf.fr/rascaline>`_ API and coding
guidelines to promote usability and interoperability with existing workflows.

Calculators return the representations as a :py:obj:`List` of :py:class:`torch.Tensor`.
We also provide a return values as a :py:class:`metatensor.TensorMap` in
:ref:`metatensor`.

.. automodule:: meshlode.calculators

.. toctree::
   :maxdepth: 1

   meshpotential
