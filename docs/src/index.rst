torch-pme
#########

.. include:: ../../README.rst
    :start-after: marker-introduction
    :end-before: marker-documentation

.. caution::

    To compute real-space short-range interactions, ``torch-pme`` requires a **neighbor
    list**, specifically the **neighbor indices** and their **distances**. We do not
    provide functionality to compute these neighbor lists, as there are already
    highly efficient libraries, such as `vesin`_, that specialize in this task and also
    support auto-differentiation.

.. _`vesin`: https://luthaf.fr/vesin

.. toctree::
   :hidden:

   installation
   examples/index
   references/index
