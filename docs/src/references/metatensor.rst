.. _metatensor:

Metatensor Bindings
###################

torch-pme calculators returning representations as :py:class:`metatensor.TensorMap`.
For using these bindings you need to install the ``metatensor.torch`` optional
dependencies.

.. code-block:: bash

   pip install .[metatensor]

For a plain :py:class:`torch.Tensor` refer to :ref:`calculators`.

Implemented Calculators
-----------------------

.. autoclass:: torchpme.metatensor.PMECalculator
    :members: forward
    :undoc-members:

.. autoclass:: torchpme.metatensor.EwaldCalculator
    :members: forward
    :undoc-members:
