Tune PME
#########

The tuning is based on the following error formulas:

.. math::
      \Delta F_\mathrm{real}
        \approx \frac{Q^2}{\sqrt{N}}
                \frac{2}{\sqrt{r_{\text{cutoff}} V}}
                e^{-r_{\text{cutoff}}^2 / 2 \sigma^2}

.. math::
    \Delta F_\mathrm{Fourier}^\mathrm{PME}
        \approx 2\pi^{1/4}\sqrt{\frac{3\sqrt{2} / \sigma}{N(2p+3)}}
                \frac{Q^2}{L^2}\frac{(\sqrt{2}H/\sigma)^{p+1}}{(p+1)!} \times
                \exp{\frac{(p+1)[\log{(p+1)} - \log 2 - 1]}{2}} \left< \phi_p^2 \right> ^{1/2}

where :math:`N` is the number of charges, :math:`Q^2 = \sum_{i = 1}^N q_i^2`, is the sum of squared
charges, :math:`r_{\text{cutoff}}` is the short-range cutoff, :math:`V` is the volume of the
simulation box, :math:`p` is the order of the interpolation scheme, :math:`H` is the spacing of mesh
points, and :math:`\phi_p^2 = H^{-(p+1)}\prod_{s\in S_H^{(p)}}(x - s)`, in which :math:`S_H^{(p)}` is
the :math:`p+1` mesh points closest to the point :math:`x`.

.. autofunction:: torchpme.tuning.pme.tune_pme

.. autoclass:: torchpme.tuning.pme.PMEErrorBounds
    :members:
