.. _calculators:

Calculators
###########

Calculators evaluate range-separated models of non-bonded interactions. The most
fundamental classes that provide the interface to compute these types of interactions
are *calculators*, that take information on the periodic cell and the atom types and
positions and computes interatomic potential-like terms.

Our calculator API follows the `rascaline <https://luthaf.fr/rascaline>`_ API and coding
guidelines to promote usability and interoperability with existing workflows. All
calculators return the representations as a :py:obj:`List` of :py:class:`torch.Tensor`.

We also provide a return values as a :py:class:`metatensor.TensorMap` in
:ref:`metatensor`.

.. note::

   All calculators compute the potential, from which the "atom-wise" energy :math:`E_i`
   can be determined by multiplying the potential :math:`V_i` with the charges :math:`q`
   . The total electrostatic energy :math:`E` is then the sum of all :math:`E_i`.

   .. math::

      E_i = q_i \cdot V_i

Implemented Calculators
-----------------------

.. toctree::
   :maxdepth: 1
   :glob:

   ./*
