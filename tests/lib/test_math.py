import numpy as np
import torch
from scipy.special import exp1

from torchpme.lib import exp1 as torch_exp1


def finite_difference_derivative(func, x, h=1e-5):
    return (func(x + h) - func(x - h)) / (2 * h)


def test_torch_exp1_consistency_with_scipy():
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random_tensor = torch.rand(100000) * 1000
    random_array = random_tensor.numpy()
    scipy_result = exp1(random_array)
    torch_result = torch_exp1(random_tensor)
    assert np.allclose(scipy_result, torch_result.numpy(), atol=1e-15)


def test_torch_exp1_derivative():
    x = torch.rand(1, dtype=torch.float64, requires_grad=True)
    torch_result = torch_exp1(x)
    torch_result.backward()
    torch_exp1_prime = x.grad
    finite_diff_result = finite_difference_derivative(exp1, x.detach().numpy())
    assert np.allclose(torch_exp1_prime.numpy(), finite_diff_result, atol=1e-6)
