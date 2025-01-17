import torch
from scipy.special import exp1
from torch.special import gammaln


def gamma(x: torch.Tensor) -> torch.Tensor:
    """
    (Complete) Gamma function.

    pytorch has not implemented the commonly used (complete) Gamma function. We define
    it in a custom way to make autograd work as in
    https://discuss.pytorch.org/t/is-there-a-gamma-function-in-pytorch/17122
    """
    return torch.exp(gammaln(x))


class CustomExp1(torch.autograd.Function):
    """Custom exponential integral function Exp1(x) to have an autograd-compatible version."""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        input_numpy = input.cpu().numpy() if not input.is_cpu else input.numpy()
        return torch.tensor(exp1(input_numpy), device=input.device, dtype=input.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        return -grad_output * torch.exp(-input) / input


def torch_exp1(input):
    """Wrapper for the custom exponential integral function."""
    return CustomExp1.apply(input)


def gammaincc_over_powerlaw(exponent: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Function to compute the regularized incomplete gamma function complement for integer exponents."""
    if exponent == 1:
        return torch.exp(-z) / z
    if exponent == 2:
        return torch.sqrt(torch.pi / z) * torch.erfc(torch.sqrt(z))
    if exponent == 3:
        return torch_exp1(z)
    if exponent == 4:
        return 2 * (
            torch.exp(-z) - torch.sqrt(torch.pi * z) * torch.erfc(torch.sqrt(z))
        )
    if exponent == 5:
        return torch.exp(-z) - z * torch_exp1(z)
    if exponent == 6:
        return (
            (2 - 4 * z) * torch.exp(-z)
            + 4 * torch.sqrt(torch.pi * z**3) * torch.erfc(torch.sqrt(z))
        ) / 3
    raise ValueError(f"Unsupported exponent: {exponent}")
