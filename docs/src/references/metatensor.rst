.. _metatensor:

Metatensor Bindings
###################

torch-pme calculators returning representations as :class:`metatensor.TensorMap`.
For using these bindings you need to install the ``metatensor.torch`` optional
dependencies.

.. code-block:: bash

   pip install .[metatensor]

For a plain :class:`torch.Tensor` refer to :ref:`calculators`.

For evaluating many systems in a single call through this interface, see the
:ref:`batched evaluation with tiling <batched-tiling>` page, which also documents
:func:`torchpme.metatensor.prepare_tiled_batch`.

Implemented Calculators
-----------------------

.. autoclass:: torchpme.metatensor.Calculator
    :members:

.. autoclass:: torchpme.metatensor.EwaldCalculator
    :members: forward, forward_batched

.. autoclass:: torchpme.metatensor.P3MCalculator
    :members: forward

.. autoclass:: torchpme.metatensor.PMECalculator
    :members: forward

Examples using the Metatensor Bindings
--------------------------------------

.. minigallery::

    torchpme.metatensor.Calculator
    torchpme.metatensor.EwaldCalculator
