from typing import Optional

import torch


class CubicSpline(torch.nn.Module):
    r"""
    Cubic spline calculator.

    Class implementing a cubic spline for a real-valued function.

    :param x_points:  Abscissas of the splining points for the real-space function
    :param y_points:  Ordinates of the splining points for the real-space function
    """

    def __init__(self, x_points: torch.Tensor, y_points: torch.Tensor):
        super().__init__()

        # stores grid information needed to compute the
        self.x_points = x_points
        self.y_points = y_points
        self.d2y_points = compute_second_derivatives(x_points, y_points)
        self._intervals = self.x_points[1:] - self.x_points[:-1]
        self._h2over6 = self._intervals**2 / 6

    def forward(self, x: torch.Tensor):
        """
        Evaluates the spline at the points provided.

        :param x: One or more positions to evaluate the splined function.
        """
        # Calculate the spline for each x
        i = torch.searchsorted(self.x_points, x, right=True) - 1
        i = torch.clamp(i, 0, len(self.x_points) - 2)

        h = self._intervals[i]
        a = (self.x_points[i + 1] - x) / h
        b = (x - self.x_points[i]) / h
        h2over6 = self._h2over6[i]
        return a * (
            self.y_points[i] + (a * a - 1) * self.d2y_points[i] * h2over6
        ) + b * (self.y_points[i + 1] + (b * b - 1) * self.d2y_points[i + 1] * h2over6)


class CubicSplineLongRange(CubicSpline):
    r"""
    Inverse-axis cubic spline calculator.

    Computes a spline on a :math:`1/x` grid, "extending" it so that
    it converges smoothly to zero as :math:`x\rightarrow\infty`.

    :param x_points:  Abscissas of the splining points for the real-space function
    :param y_points:  Ordinates of the splining points for the real-space function
    """

    def __init__(self, x_points: torch.Tensor, y_points: torch.Tensor):
        # compute on a inverse grid
        ix_points = torch.cat(
            [
                torch.zeros((1,), dtype=x_points.dtype, device=x_points.device),
                torch.reciprocal(torch.flip(x_points, dims=[0])),
            ],
            dim=0,
        )
        iy_points = torch.cat(
            [
                torch.zeros((1,), dtype=x_points.dtype, device=x_points.device),
                torch.flip(y_points, dims=[0]),
            ],
            dim=0,
        )
        super().__init__(ix_points, iy_points)

    def forward(self, x: torch.Tensor):
        return super().forward(torch.reciprocal(x))


def _solve_tridiagonal(a, b, c, d):
    """
    Helper function to solve a tri-diagonal problem
    """

    n = len(d)
    # Create copies to avoid modifying the original arrays
    c_prime = torch.zeros(n)
    d_prime = torch.zeros(n)

    # Initial coefficients
    c_prime[0] = c[0] / b[0]
    d_prime[0] = d[0] / b[0]

    # Forward sweep
    for i in range(1, n):
        denom = b[i] - a[i] * c_prime[i - 1]
        c_prime[i] = c[i] / denom if i < n - 1 else 0
        d_prime[i] = (d[i] - a[i] * d_prime[i - 1]) / denom

    # Backward substitution
    x = torch.zeros(n)
    x[-1] = d_prime[-1]
    for i in reversed(range(n - 1)):
        x[i] = d_prime[i] - c_prime[i] * x[i + 1]
    return x


def compute_second_derivatives(
    x_points: torch.Tensor, y_points: torch.Tensor, high_accuracy: Optional[bool] = True
):
    """
    Computes second derivatives given the grid points of a cubic spline.

    :param x_points:  Abscissas of the splining points for the real-space function
    :param y_points:  Ordinates of the splining points for the real-space function
    :param high_accuracy: bool, perform calculation in double precision

    :return:  The second derivatives for the spline points
    """

    # Do the calculation in float64 if required
    x = x_points
    y = y_points
    if high_accuracy:
        x = x.to(dtype=torch.float64)
        y = y.to(dtype=torch.float64)

    # Calculate intervals
    intervals = x[1:] - x[:-1]
    dy = (y[1:] - y[:-1]) / intervals

    # Create zero boundary conditions (natural spline)
    d2y = torch.zeros_like(x, dtype=torch.float64)

    n = len(x)
    a = torch.zeros(n)  # Sub-diagonal (a[1..n-1])
    b = torch.zeros(n)  # Main diagonal (b[0..n-1])
    c = torch.zeros(n)  # Super-diagonal (c[0..n-2])
    d = torch.zeros(n)  # Right-hand side (d[0..n-1])

    # Natural spline boundary conditions
    b[0] = 1
    d[0] = 0  # Second derivative at the first point is zero
    b[-1] = 1
    d[-1] = 0  # Second derivative at the last point is zero

    # Fill the diagonals and right-hand side
    for i in range(1, n - 1):
        a[i] = intervals[i - 1] / 6
        b[i] = (intervals[i - 1] + intervals[i]) / 3
        c[i] = intervals[i] / 6
        d[i] = dy[i] - dy[i - 1]

    """
    # Create matrix A and vector B for solving the system A * d2y = B
    n = len(x)
    A = torch.zeros((n, n))
    B = torch.zeros(n)

    A[0, 0] = 1  # Natural spline condition at the first point
    A[-1, -1] = 1  # Natural spline condition at the last point

    for i in range(1, n - 1):
        A[i, i - 1] = intervals[i - 1] / 6
        A[i, i] = (intervals[i - 1] + intervals[i]) / 3
        A[i, i + 1] = intervals[i] / 6
        B[i] = dy[i] - dy[i - 1]

    # Solve the system to find the second derivatives
    d2y = torch.linalg.solve(A, B)
    """
    d2y = _solve_tridiagonal(a, b, c, d)

    # Converts back to the original dtype
    return d2y.to(x_points.dtype)


def compute_spline_ft(
    k_points: torch.Tensor,
    x_points: torch.Tensor,
    y_points: torch.Tensor,
    d2y_points: torch.Tensor,
    high_precision: Optional[bool] = True,
):
    r"""
    Computes the Fourier transform of a splined radial function.

    Evaluates the integral

    .. math::
        \hat{f}(k) =4\pi\int \mathrm{d}r \frac{\sin k r}{k} r f(r)

    where :math:`f(r)` is expressed as a cubic spline. The function
    also includes a tail correction to continue the integral beyond
    the last splined point, assuming that the function converges to
    zero at infinity.

    :param k_points:  Points on which the Fourier kernel should be
        computed. It is a good idea to take them to be
        :math:`2\pi/x` based on the real-space ``x_points``
    :param x_points:  Abscissas of the splining points for the real-space function
    :param y_points:  Ordinates of the splining points for the real-space function
    :param d2y_points:  Second derivatives for the spline points
    :param high_accuracy: bool, perform calculation in double precision

    :return: The radial Fourier transform :math:`\hat{f}(k)` computed
        at the ``k_points`` provided.
    """

    # chooses precision for the FT evaluation
    dtype = x_points.dtype
    if high_precision:
        dtype = torch.float64

    # broadcast to compute at once on all k values.
    # all these are terms that enter the analytical integral.
    # might be possible to write this in a more concise way, but
    # this works and is reasonably numerically stable, so it will do
    k = k_points.reshape(-1, 1).to(dtype)
    ri = x_points[torch.newaxis, :-1].to(dtype)
    yi = y_points[torch.newaxis, :-1].to(dtype)
    d2yi = d2y_points[torch.newaxis, :-1].to(dtype)
    dr = (x_points[torch.newaxis, 1:] - x_points[torch.newaxis, :-1]).to(dtype)
    dy = (y_points[torch.newaxis, 1:] - y_points[torch.newaxis, :-1]).to(dtype)
    dd2y = (d2y_points[torch.newaxis, 1:] - d2y_points[torch.newaxis, :-1]).to(dtype)
    coskx = torch.cos(k * ri)
    sinkx = torch.sin(k * ri)
    # cos r+dr - cos r
    dcoskx = 2 * torch.sin(k * dr / 2) * torch.sin(k * (dr / 2 + ri))
    # sin r+dr - cos r
    dsinkx = -2 * torch.sin(k * dr / 2) * torch.cos(k * (dr / 2 + ri))

    # this monstruous expression computes, for each interval in the spline,
    # \int_{r_i}^{r_{i+1}} .... using the coefficients of the spline.
    # the expression here is also cast in a Horner form, and uses a few
    # tricks to make it stabler, as a naive implementation is very noisy
    # in float32 for small k. for instance, the first term contains the difference
    # of two cosines, but is computed with a trigonometric identity
    # (see the definition of dcoskx) to avoid the 1-k^2 form of the bare cosines
    ft_interval = 24 * dcoskx * dd2y + k * (
        6 * dsinkx * (3 * d2yi * dr + dd2y * (4 * dr + ri))
        - 24 * dd2y * dr * sinkx
        + k
        * (
            6 * coskx * dr * (3 * d2yi * dr + dd2y * (2 * dr + ri))
            - 2
            * dcoskx
            * (6 * dy + dr * ((6 * d2yi + 5 * dd2y) * dr + 3 * (d2yi + dd2y) * ri))
            + k
            * (
                dr
                * (
                    12 * dy
                    + 3 * d2yi * dr * (dr + 2 * ri)
                    + dd2y * dr * (2 * dr + 3 * ri)
                )
                * sinkx
                + dsinkx
                * (
                    -6 * dy * ri
                    - 3 * d2yi * dr**2 * (dr + ri)
                    - 2 * dd2y * dr**2 * (dr + ri)
                    - 6 * dr * (2 * dy + yi)
                )
                + k
                * (
                    6 * dcoskx * dr * (dr + ri) * (dy + yi)
                    + coskx * (6 * dr * ri * yi - 6 * dr * (dr + ri) * (dy + yi))
                )
            )
        )
    )

    # especially for Coulomb-like integrals, no matter how far we push the splining
    # in real space, the tail matters, so we compute it separately. to do this
    # stably and acurately, we build the tail as a spline in 1/r (using the last two)
    # points of the spline) and use an analytical expression for the resulting
    # integral from the last point to infinity

    tail_d2y = compute_second_derivatives(
        torch.tensor([0, 1 / x_points[-1], 1 / x_points[-2]]),
        torch.tensor([0, y_points[-1], y_points[-2]]),
    )

    r0 = x_points[-1]
    y0 = y_points[-1]
    d2y0 = tail_d2y[1]
    # the expression contains the cosine integral special function, that
    # is only available in scipy
    try:
        from scipy.special import sici
    except ImportError as err:
        raise ImportError(
            "Computing the Fourier-domain kernel based on a spline requires scipy"
        ) from err

    # to numpy and back again
    cosint = torch.from_numpy(sici((k * r0).detach().cpu().numpy())[1]).to(
        dtype=dr.dtype, device=dr.device
    )

    tail = (
        -2
        * torch.pi
        * (
            (d2y0 - 6 * r0**2 * y0) * torch.cos(k * r0)
            + d2y0 * k * r0 * (k * r0 * cosint - torch.sin(k * r0))
        )
    ) / (3.0 * k**2 * r0)

    ft_full = (
        2 * torch.pi / 3 * torch.sum(ft_interval / dr, axis=1).reshape(-1, 1) / k**6
        + tail
    )

    return ft_full.reshape(k_points.shape).to(k_points.dtype)
