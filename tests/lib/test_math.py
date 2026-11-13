import numpy as np
import scipy.special
import torch

from torchpme.lib import exp1


def test_torch_exp1_consistency_with_scipy():
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random_tensor = torch.rand(100000) * 1000
    random_array = random_tensor.numpy()
    scipy_result = scipy.special.exp1(random_array)
    torch_result = exp1(random_tensor)
    assert np.allclose(scipy_result, torch_result.numpy(), atol=1e-15)


def test_torch_exp1_derivative():
    # spans both the series (x <= 1) and the continued fraction (x > 1) branches
    x = torch.logspace(-6, 2, 1000, dtype=torch.float64, requires_grad=True)
    exp1(x).sum().backward()
    expected = -torch.exp(-x.detach()) / x.detach()
    torch.testing.assert_close(x.grad, expected, rtol=1e-12, atol=0.0)


def test_torch_exp1_second_derivative():
    x = torch.logspace(-6, 2, 1000, dtype=torch.float64, requires_grad=True)
    (grad,) = torch.autograd.grad(exp1(x).sum(), x, create_graph=True)
    grad.sum().backward()
    xd = x.detach()
    expected = torch.exp(-xd) * (xd + 1) / xd**2
    torch.testing.assert_close(x.grad, expected, rtol=1e-12, atol=0.0)
