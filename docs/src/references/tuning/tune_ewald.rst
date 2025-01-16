Tune Ewald
##########

The tuning is based on the following error formulas:

.. math::
      \Delta F_\mathrm{real}
        \approx \frac{Q^2}{\sqrt{N}}
                \frac{2}{\sqrt{r_{\text{cutoff}} V}}
                e^{-r_{\text{cutoff}}^2 / 2 \sigma^2}

.. math::
    \Delta F_\mathrm{Fourier}^\mathrm{Ewald}
        \approx \frac{Q^2}{\sqrt{N}}
                \frac{\sqrt{2} / \sigma}{\pi\sqrt{2 V / h}} e^{-2\pi^2 \sigma^2 / h ^ 2}

where :math:`N` is the number of charges, :math:`Q^2 = \sum_{i = 1}^N q_i^2`, is the sum of squared
charges, :math:`r_{\text{cutoff}}` is the short-range cutoff, :math:`V` is the volume of the
simulation box and :math:`h^2` is the long range wavelength.

.. autofunction:: torchpme.tuning.ewald.tune_ewald

.. autoclass:: torchpme.tuning.ewald.EwaldErrorBounds
    :members:
