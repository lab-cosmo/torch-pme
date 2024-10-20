import pytest
import torch
from torch.testing import assert_close

from torchpme.lib.potentials import CoulombPotential, SplinePotential
from torchpme.lib.splines import CubicSpline, CubicSplineLongRange


@pytest.mark.parametrize("function", [torch.exp, torch.sin, torch.tanh])
def test_spline_function(function):
    x_grid = torch.linspace(-4, 4, 200)
    x_test = torch.linspace(-torch.pi, torch.pi, 23)
    y_grid = function(x_grid)
    y_test = function(x_test)

    spline_test = CubicSpline(x_grid, y_grid)
    z_grid = spline_test(x_grid)

    # checks that the spline interpolates exactly on the grid
    assert_close(y_grid, z_grid, atol=1e-15, rtol=0.0)

    # checks that the spline is accurate-ish elsewhere
    z_test = spline_test(x_test)
    assert_close(y_test, z_test, atol=1e-6, rtol=0.0)


@pytest.mark.parametrize("function", [torch.log, torch.reciprocal])
def test_inverse_spline(function):
    x_grid = torch.logspace(-4, 4, 500)
    x_test = torch.logspace(-torch.pi, torch.pi, 23)
    y_grid = function(x_grid)

    spline_test = CubicSpline(x_grid, y_grid)
    lr_spline_test = CubicSplineLongRange(x_grid, y_grid)
    z_grid = spline_test(x_grid)
    lr_grid = lr_spline_test(x_grid)

    # on the grid points, the two splines should be exact
    assert_close(lr_grid, z_grid, atol=1e-15, rtol=0.0)

    # checks that the spline is accurate-ish elsewhere
    z_test = spline_test(x_test)
    lr_test = lr_spline_test(x_test)
    assert_close(lr_test, z_test, atol=3e-4, rtol=1e-5)

    # checks that the lr version extrapolates well
    # towards infinity (it should be nearly perfect for reciprocal)
    if function is torch.reciprocal:
        x_grid = torch.logspace(4, 10, 100)
        y_grid = function(x_grid)
        z_grid = spline_test(x_grid)
        lr_grid = lr_spline_test(x_grid)
        assert_close(lr_grid, y_grid, atol=1e-8, rtol=1e-9)


def test_spline_potential():
    coulomb = CoulombPotential(smearing=1.0)
    x_grid = torch.logspace(-4, 4, 500)
    y_grid = coulomb.lr_from_dist(x_grid)

    spline = SplinePotential(r_grid=x_grid, y_grid_lr=y_grid)
    t_grid = torch.logspace(-torch.pi, torch.pi, 100)
    z_coul = coulomb.lr_from_dist(t_grid)
    z_spline = spline.lr_from_dist(t_grid)
    assert_close(z_coul, z_spline, atol=1e-5, rtol=0)

    k_grid2 = torch.logspace(-2, 2, 500) ** 2
    krn_coul = coulomb.kernel_from_k_sq(k_grid2)
    krn_spline = spline.kernel_from_k_sq(k_grid2)
    print("kernel values", krn_coul, krn_spline)
    assert_close(krn_coul, krn_spline, atol=1e-5, rtol=0)
