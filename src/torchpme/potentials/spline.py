from typing import Optional

import torch

from ..utils.splines import (
    CubicSpline,
    CubicSplineReciprocal,
    compute_second_derivatives,
    compute_spline_ft,
)
from .potential import Potential


class SplinePotential(Potential):
    r"""
    Potential built from a spline interpolation.

    The potential is assumed to have only a long-range part, but one can also
    add a short-range part if needed, by inheriting and redefining
    ``sr_from_dist``.
    The real-space potential is computed based on a cubic spline built at
    initialization time. The Fourier-domain kernel is computed numerically
    as a spline, too.  Assumes the infinite-separation value of the
    potential to be zero.

    :param r_grid: radial grid for the real-space evaluation
    :param y_grid: potential values for the real-space evaluation
    :param k_grid: radial grid for the k-space evaluation;
        computed automatically from ``r_grid`` if absent.
    :param yhat_grid: potential values for the k-space evaluation;
        computed automatically from ``y_grid`` if absent.
    :param reciprocal: flag that determines if the splining should
        be performed on a :math:`1/r` axis; suitable to describe
        long-range potentials. ``r_grid`` should contain only
        stricty positive values.
    :param y_at_zero: value to be used for :math:`r\rightarrow 0`
        when using a reciprocal spline
    :param yhat_at_zero: value to be used for :math:`k\rightarrow 0`
        in the k-space kernel
    :param smearing: The length scale associated with the switching between
        :math:`V_{\mathrm{SR}}(r)` and :math:`V_{\mathrm{LR}}(r)`
    :param exclusion_radius: A length scale that defines a *local environment* within
        which the potential should be smoothly zeroed out, as it will be described by a
        separate model.
    :param dtype: type used for the internal buffers and parameters
    :param device: device used for the internal buffers and parameters
    """

    def __init__(
        self,
        r_grid: torch.Tensor,
        y_grid: torch.Tensor,
        k_grid: Optional[torch.Tensor] = None,
        yhat_grid: Optional[torch.Tensor] = None,
        reciprocal: Optional[bool] = False,
        y_at_zero: Optional[float] = None,
        yhat_at_zero: Optional[float] = None,
        smearing: Optional[float] = None,
        exclusion_radius: Optional[float] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__(
            smearing=smearing,
            exclusion_radius=exclusion_radius,
            dtype=dtype,
            device=device,
        )
        if dtype is None:
            dtype = torch.get_default_dtype()
        if device is None:
            device = torch.device("cpu")

        if len(y_grid) != len(r_grid):
            raise ValueError("Length of radial grid and value array mismatch.")

        if reciprocal:
            if torch.min(r_grid) <= 0.0:
                raise ValueError(
                    "Positive-valued radial grid is needed for reciprocal axis spline."
                )
            self._spline = CubicSplineReciprocal(r_grid, y_grid, y_at_zero=y_at_zero)
        else:
            self._spline = CubicSpline(r_grid, y_grid)

        if k_grid is None:
            # defaults to 2pi/r_grid_points if reciprocal, to r_grid if not
            if reciprocal:
                k_grid = torch.pi * 2 * torch.reciprocal(r_grid).flip(dims=[0])
            else:
                k_grid = r_grid.clone()

        if yhat_grid is None:
            # computes automatically!
            yhat_grid = compute_spline_ft(
                k_grid,
                r_grid,
                y_grid,
                compute_second_derivatives(r_grid, y_grid),
            )

        # the function is defined for k**2, so we define the grid accordingly
        if reciprocal:
            self._krn_spline = CubicSplineReciprocal(
                k_grid**2, yhat_grid, y_at_zero=yhat_at_zero
            )
        else:
            self._krn_spline = CubicSpline(k_grid**2, yhat_grid)

        if y_at_zero is None:
            self._y_at_zero = self._spline(torch.tensor([0.0]))
        else:
            self._y_at_zero = y_at_zero

        if yhat_at_zero is None:
            self._yhat_at_zero = self._krn_spline(torch.tensor([0.0]))
        else:
            self._yhat_at_zero = yhat_at_zero

    def from_dist(self, dist: torch.Tensor) -> torch.Tensor:
        # if the full spline is not given, falls back on the lr part
        return self.lr_from_dist(dist) + self.sr_from_dist(dist)

    def sr_from_dist(self, dist: torch.Tensor) -> torch.Tensor:
        """
        Short-range part of the range-separated potential.

        :param dist: torch.tensor containing the distances at which the potential is to
            be evaluated.
        """
        return 0.0 * dist

    def lr_from_dist(self, dist: torch.Tensor) -> torch.Tensor:
        return self._spline(dist)

    def lr_from_k_sq(self, k_sq: torch.Tensor) -> torch.Tensor:
        return self._krn_spline(k_sq)

    def self_contribution(self) -> torch.Tensor:
        return self._y_at_zero

    def background_correction(self) -> torch.Tensor:
        return torch.tensor([0.0])

    from_dist.__doc__ = Potential.from_dist.__doc__
    lr_from_dist.__doc__ = Potential.lr_from_dist.__doc__
    lr_from_k_sq.__doc__ = Potential.lr_from_k_sq.__doc__
    self_contribution.__doc__ = Potential.self_contribution.__doc__
    background_correction.__doc__ = Potential.background_correction.__doc__
