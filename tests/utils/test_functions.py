import numpy as np
import torch
from scipy.special import exp1

from torchpme.potentials.inversepowerlaw import torch_exp1


def finite_difference_derivative(func, x, h=1e-5):
    return (func(x + h) - func(x - h)) / (2 * h)


def test_torch_exp1_consistency_with_scipy():
    x = torch.rand(1000, dtype=torch.float64)
    torch_result = torch_exp1(x)
    scipy_result = exp1(x.numpy())
    assert np.allclose(torch_result.numpy(), scipy_result, atol=1e-6)


def test_torch_exp1_derivative():
    x = torch.rand(1, dtype=torch.float64, requires_grad=True)
    torch_result = torch_exp1(x)
    torch_result.backward()
    torch_exp1_prime = x.grad
    finite_diff_result = finite_difference_derivative(exp1, x.detach().numpy())
    assert np.allclose(torch_exp1_prime.numpy(), finite_diff_result, atol=1e-6)
