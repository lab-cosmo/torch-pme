torch-pme
#########

.. include:: ../../README.rst
    :start-after: marker-introduction
    :end-before: marker-documentation

To compute real-space short-range interactions, ``torch-pme`` requires neighbor
information, such as the **neighbor indices** and their **distances**. We do not provide
functionality to compute these neighbor lists, as there are already highly efficient
libraries, such as `vesin`_, that are specialized in this task and also support
auto-differentiation.

.. _`vesin`: https://luthaf.fr/vesin

.. toctree::
   :hidden:

   installation
   examples/index
   references/index
