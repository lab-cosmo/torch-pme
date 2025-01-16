Base Classes
############

Current scheme behind all tuning functions is grid-searching based, focusing on the Fourier
space parameters like ``lr_wavelength``, ``mesh_spacing`` and ``interpolation_nodes``.
For real space parameter ``cutoff``, it is treated as a hyperparameter here, which
should be manually specified by the user. The parameter ``smearing`` is determined by
the real space error formula and is set to achieve a real space error of
``desired_accuracy / 4``.

The Fourier space parameters are all discrete, so it's convenient to do the grid-search.
Default searching-ranges are provided for those parameters. For ``lr_wavelength``, the
values are chosen to be with a minimum of 1 and a maximum of 13 mesh points in each
spatial direction ``(x, y, z)``. For ``mesh_spacing``, the values are set to have
minimally 2 and maximally 7 mesh points in each spatial direction, for both the P3M and
PME method. The values of ``interpolation_nodes`` are the same as those supported in
:class:`torchpme.lib.MeshInterpolator`.

In the grid-searching, all possible parameter combinations are evaluated. The error
associated with the parameter is estimated by the error formulas implemented in the
subclasses of :class:`torchpme.tuning.tuner.TuningErrorBounds`. Parameter with
the error within the desired accuracy are benchmarked for computational time by
:class:`torchpme.tuning.tuner.TuningTimings` The timing of the other parameters are
not tested and set to infinity.

The return of these tuning functions contains the ``smearing`` and a dictionary, in
which there is parameter for the Fourier space. The parameter is that of the desired
accuracy and the shortest timing. The parameter of the smallest error will be returned
in the case that no parameter can fulfill the accuracy requirement.


.. autoclass:: torchpme.tuning.tuner.TunerBase
    :members:

.. autoclass:: torchpme.tuning.tuner.GridSearchTuner
    :members:

.. autoclass:: torchpme.tuning.tuner.TuningTimings
    :members:

.. autoclass:: torchpme.tuning.tuner.TuningErrorBounds
    :members:

