Tune P3M
#########

The tuning is based on the following error formulas:

.. math::
      \Delta F_\mathrm{real}
        \approx \frac{Q^2}{\sqrt{N}}
                \frac{2}{\sqrt{r_{\text{cutoff}} V}}
                e^{-r_{\text{cutoff}}^2 / 2 \sigma^2}

.. math::
    \Delta F_\mathrm{Fourier}^\mathrm{P3M}
        \approx \frac{Q^2}{L^2}(\frac{\sqrt{2}H}{\sigma})^p
                    \sqrt{\frac{\sqrt{2}L}{N\sigma}
                    \sqrt{2\pi}\sum_{m=0}^{p-1}a_m^{(p)}(\frac{\sqrt{2}H}{\sigma})^{2m}}

where :math:`N` is the number of charges, :math:`Q^2 = \sum_{i = 1}^N q_i^2`, is the sum of squared
charges, :math:`r_{\text{cutoff}}` is the short-range cutoff, :math:`V` is the volume of the
simulation box, :math:`p` is the order of the interpolation scheme, :math:`H` is the spacing of mesh
points and :math:`a_m^{(p)}` is an expansion coefficient.


.. autofunction:: torchpme.tuning.p3m.tune_p3m

.. autoclass:: torchpme.tuning.p3m.P3MErrorBounds
    :members:
