.. _metatensor:

Metatensor Bindings
###################

torch-pme calculators returning representations as :class:`metatensor.TensorMap`.
For using these bindings you need to install the ``metatensor.torch`` optional
dependencies.

.. code-block:: bash

   pip install .[metatensor]

For a plain :class:`torch.Tensor` refer to :ref:`calculators`.

Implemented Calculators
-----------------------

.. autoclass:: torchpme.metatensor.Calculator
    :members:

.. autoclass:: torchpme.metatensor.EwaldCalculator
    :members: forward

.. autoclass:: torchpme.metatensor.P3MCalculator
    :members: forward

.. autoclass:: torchpme.metatensor.PMECalculator
    :members: forward

Examples using the Metatensor Bindings
--------------------------------------

.. minigallery::

    torchpme.metatensor.Calculator
    torchpme.metatensor.PMECalculator
    torchpme.metatensor.P3MCalculator
    torchpme.metatensor.EwaldCalculator
