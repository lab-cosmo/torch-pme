import pytest
import torch
from torch.testing import assert_close

from torchpme.lib.splines import CubicSpline


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
