import torch
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
    """
    Compute the exponential integral E1(x) for x > 0.
    :param input: Input tensor (x > 0)
    :return: Exponential integral E1(x)
    """

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)

        # Constants
        SCIPY_EULER = (
            0.577215664901532860606512090082402431  # Euler-Mascheroni constant
        )
        inf = torch.inf

        # Handle case when x == 0
        result = torch.full_like(input, inf)
        mask = input > 0

        # Compute for x <= 1
        x_small = input[mask & (input <= 1)]
        if x_small.numel() > 0:
            e1 = torch.ones_like(x_small)
            r = torch.ones_like(x_small)
            for k in range(1, 26):
                r = -r * k * x_small / (k + 1.0) ** 2
                e1 += r
                if torch.all(torch.abs(r) <= torch.abs(e1) * 1e-15):
                    break
            result[mask & (input <= 1)] = (
                -SCIPY_EULER - torch.log(x_small) + x_small * e1
            )

        # Compute for x > 1
        x_large = input[mask & (input > 1)]
        if x_large.numel() > 0:
            m = 20 + (80.0 / x_large).to(torch.int32)
            t0 = torch.zeros_like(x_large)
            for k in range(m.max(), 0, -1):
                t0 = k / (1.0 + k / (x_large + t0))
            t = 1.0 / (x_large + t0)
            result[mask & (input > 1)] = torch.exp(-x_large) * t

        return result

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        return -grad_output * torch.exp(-input) / input


def exp1(input):
    """Wrapper for the custom exponential integral function."""
    return CustomExp1.apply(input)


def gammaincc_over_powerlaw(exponent: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """
    Function to compute the regularized incomplete gamma function complement for integer
    exponents.
    param exponent: Exponent of the power law
    param z: Value at which to evaluate the function
    return: Regularized incomplete gamma function complement
    """
    if exponent == 1:
        return torch.exp(-z) / z
    if exponent == 2:
        return torch.sqrt(torch.pi / z) * torch.erfc(torch.sqrt(z))
    if exponent == 3:
        return exp1(z)
    if exponent == 4:
        return 2 * (
            torch.exp(-z) - torch.sqrt(torch.pi * z) * torch.erfc(torch.sqrt(z))
        )
    if exponent == 5:
        return torch.exp(-z) - z * exp1(z)
    if exponent == 6:
        return (
            (2 - 4 * z) * torch.exp(-z)
            + 4 * torch.sqrt(torch.pi * z**3) * torch.erfc(torch.sqrt(z))
        ) / 3
    raise ValueError(f"Unsupported exponent: {exponent}")
