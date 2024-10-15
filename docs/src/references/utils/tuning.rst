Tuning
######

Here we offer tuning methods for :py:class:`torchpme.EwaldCalculator` and
:py:class:`torchpme.PMECalculator`. The choice of parameters of these calculators
greatly influence the accuracy of the calculation results. To help find
the parameters that meet the accuracy requirements, one can use these functions.

The scheme behind these tuning methods is gradient-based optimization, which tries to
find the minimal of the error estimation formula and stops after the error is smaller
than the given accuracy. Because these methods are gradient-based, besure to pay
attention to the ``learning_rate`` and ``max_steps`` parameter. A good choice of these
two parameters can enhance the optimization speed and performance.

.. automodule:: torchpme.utils.tuning
    :members:
    :undoc-members:
