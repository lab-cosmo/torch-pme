import pytest
import torch
from scipy.special import expi
from torch.special import erf, erfc
from torch.testing import assert_close

from torchpme.lib import CoulombPotential, InversePowerLawPotential


def gamma(x):
    x = torch.tensor(x, dtype=dtype)
    return torch.exp(torch.special.gammaln(x))


# Define precision of floating point variables
dtype = torch.float64

# Define range of exponents covering relevant special values and more general
# floating point values beyond this range. The last four of which are inspired by:
# von Klitzing constant R_K = 2.5812... * 1e4 Ohm
# Josephson constant K_J = 4.8359... * 1e9 Hz/V
# Gravitational constant G = 6.6743... * 1e-11 m3/kgs2
# Electron mass m_e = 9.1094 * 1e-31 kg
# TODO: for the moment, InversePowerLawPotential only works for exponent 0<p<3
# ps = [1.0, 2.0, 3.0, 6.0] + [0.12345, 0.54321, 2.581304, 4.835909, 6.674311, 9.109431]
ps = [1.0, 2.0, 3.0] + [0.12345, 0.54321, 2.581304]

# Define range of smearing parameters covering relevant values
smearings = [0.1, 0.5, 1.0, 1.56]

# Define realistic range of distances on which the potentials will be evaluated
dist_min = 1.41e-2
dist_max = 27.18
num_dist = 200
dists = torch.linspace(dist_min, dist_max, num_dist, dtype=dtype)
dists_sq = dists**2

# Define realistic range of wave vectors k on which the Fourier-transformed potentials
# will be evaluated
k_min = 2 * torch.pi / 50.0
k_max = 2 * torch.pi / 0.1
num_k = 200
ks = torch.linspace(k_min, k_max, num_k, dtype=dtype)
ks_sq = ks**2

# Define machine epsilon
machine_epsilon = torch.finfo(dtype).eps

# Other shortcuts
SQRT2 = torch.sqrt(torch.tensor(2.0, dtype=dtype))
PI = torch.tensor(torch.pi, dtype=dtype)


@pytest.mark.parametrize("exponent", ps)
def test_potential_from_squared_argument(exponent):
    """
    The potentials class can either compute the potential by taking the distance r or
    its square r^2 as an argument. This test makes sure that both implementations agree.
    """

    # Compute diverse potentials for this inverse power law
    ipl = InversePowerLawPotential(exponent=exponent, smearing=0.0)
    potential_from_dist = ipl.from_dist(dists)
    potential_from_dist_sq = ipl.from_dist_sq(dists_sq)

    # Test agreement between implementations taking r vs r**2 as argument
    atol = 3e-16
    rtol = 2 * machine_epsilon
    assert_close(potential_from_dist, potential_from_dist_sq, rtol=rtol, atol=atol)


@pytest.mark.parametrize("smearing", smearings)
@pytest.mark.parametrize("exponent", ps)
def test_sr_lr_split(exponent, smearing):
    """
    This test verifies the splitting 1/r^p = V_SR(r) + V_LR(r), meaning that it tests
    whether the sum of the SR and LR parts combine to the standard inverse power-law
    potential.
    """

    # Compute diverse potentials for this inverse power law
    ipl = InversePowerLawPotential(exponent=exponent, smearing=smearing)
    potential_from_dist = ipl.from_dist(dists)
    potential_sr_from_dist = ipl.sr_from_dist(dists)
    potential_lr_from_dist = ipl.lr_from_dist(dists)
    potential_from_sum = potential_sr_from_dist + potential_lr_from_dist

    # Check that the sum of the SR and LR parts is equivalent to the original 1/r^p
    # potential. Note that the relative errors get particularly large for bigger
    # interaction exponents. If only p=1 is used, rtol can be reduced to about 3.5 times
    # the machine epsilon.
    atol = 3e-16
    rtol = 3 * machine_epsilon
    assert_close(potential_from_dist, potential_from_sum, rtol=rtol, atol=atol)


@pytest.mark.parametrize("exponent", [1.0, 2.0, 3.0])
@pytest.mark.parametrize("smearing", smearings)
def test_exact_sr(exponent, smearing):
    """
    Test that the implemented formula which works for general interaction exponents p
    does indeed reduce to the correct expression for the special case of the Coulomb
    interaction (p=1) as well as p=2,3. This test covers the SR part of the potential.
    Note that the relative tolerance could be greatly reduced if the lower end of the
    distance range (the variable dist_min) is increased, since the potential has a
    (removable) singularity at r=0.
    """

    # Compute SR part of Coulomb potential using the potentials class working for any
    # exponent
    ipl = InversePowerLawPotential(exponent=exponent, smearing=smearing)
    potential_sr_from_dist = ipl.sr_from_dist(dists)

    # Compute exact analytical expression obtained for relevant exponents
    potential_1 = erfc(dists / SQRT2 / smearing) / dists
    potential_2 = torch.exp(-0.5 * dists_sq / smearing**2) / dists_sq
    if exponent == 1.0:
        potential_exact = potential_1
    elif exponent == 2.0:
        potential_exact = potential_2
    elif exponent == 3.0:
        prefac = SQRT2 / torch.sqrt(PI) / smearing
        potential_exact = potential_1 / dists_sq + prefac * potential_2

    # Compare results. Large tolerance due to singular division
    rtol = 10 * machine_epsilon
    atol = 3e-16
    assert_close(potential_sr_from_dist, potential_exact, rtol=rtol, atol=atol)


@pytest.mark.parametrize("exponent", [1.0, 2.0, 3.0])
@pytest.mark.parametrize("smearing", smearings)
def test_exact_lr(exponent, smearing):
    """
    Test that the implemented formula which works for general interaction exponents p
    does indeed reduce to the correct expression for the special case of the Coulomb
    interaction (p=1) as well as p=2,3. This test covers the LR part of the potential.
    Note that the relative tolerance could be greatly reduced if the lower end of the
    distance range (the variable dist_min) is increased, since the potential has a
    (removable) singularity at r=0.
    """

    # Compute LR part of Coulomb potential using the potentials class working for any
    # exponent
    ipl = InversePowerLawPotential(exponent=exponent, smearing=smearing)
    potential_lr_from_dist = ipl.lr_from_dist(dists)

    # Compute exact analytical expression obtained for relevant exponents
    potential_1 = erf(dists / SQRT2 / smearing) / dists
    potential_2 = torch.exp(-0.5 * dists_sq / smearing**2) / dists_sq
    if exponent == 1.0:
        potential_exact = potential_1
    elif exponent == 2.0:
        potential_exact = 1 / dists_sq - potential_2
    elif exponent == 3.0:
        prefac = SQRT2 / torch.sqrt(PI) / smearing
        potential_exact = potential_1 / dists_sq - prefac * potential_2

    # Compare results. Large tolerance due to singular division
    rtol = 1e-10
    atol = 1e-12
    assert_close(potential_lr_from_dist, potential_exact, rtol=rtol, atol=atol)


@pytest.mark.parametrize("exponent", [1.0, 2.0])
@pytest.mark.parametrize("smearing", smearings)
def test_exact_fourier(exponent, smearing):
    """
    Test that the implemented formula which works for general interaction exponents p
    does indeed reduce to the correct expression for the special case of the Coulomb
    interaction (p=1) as well as p=2,3. This test covers the Fourier-transform.
    Note that the relative tolerance could be greatly reduced if the lower end of the
    distance range (the variable dist_min) is increased, since the potential has a
    (removable) singularity at r=0.
    """

    # Compute LR part of Coulomb potential using the potentials class working for any
    # exponent
    ipl = InversePowerLawPotential(exponent=exponent, smearing=smearing)
    fourier_from_class = ipl.from_k_sq(ks_sq)

    # Compute exact analytical expression obtained for relevant exponents
    if exponent == 1.0:
        fourier_exact = 4 * PI / ks_sq * torch.exp(-0.5 * smearing**2 * ks_sq)
    elif exponent == 2.0:
        fourier_exact = 2 * PI**2 / ks * erfc(smearing * ks / SQRT2)
    elif exponent == 3.0:
        fourier_exact = -2 * PI * expi(-0.5 * smearing**2 * ks_sq)

    # Compare results. Large tolerance due to singular division
    rtol = 1e-14
    atol = 1e-14
    assert_close(fourier_from_class, fourier_exact, rtol=rtol, atol=atol)


@pytest.mark.parametrize("smearing", smearings)
@pytest.mark.parametrize("exponent", ps[:-1])  # for p=9.11, the results are unstable
def test_lr_value_at_zero(exponent, smearing):
    """
    The LR part of the potential should no longer have a singularity as r-->0. Instead,
    the value of the potential should converge to an analytical expression that depends
    on both the exponent p and smearing sigma,namely
    V_LR(0) = Gamma((p+2)/2) / (2*sigma**2)**(p/2)

    Note that in general, V_LR as r-->0 is a limit of the form 0/0, and hence
    numerically unstable. This issue is more severe for exponents p that are large,
    which is why the biggest exponent is excluded from this test. By restricting to even
    smaller values of p, one could set the tolerance in this test to an even lower
    value.

    In practice, this should not be such an issue since no two atoms should approach
    each other until their distance is 1e-5 (the value used here).
    """
    # Get atomic density at tiny distance
    dist_small = torch.tensor(1e-8, dtype=dtype)
    ipl = InversePowerLawPotential(exponent=exponent, smearing=smearing)
    potential_close_to_zero = ipl.lr_from_dist(dist_small)

    # Compare to
    exact_value = 1.0 / (2 * smearing**2) ** (exponent / 2) / gamma(exponent / 2 + 1.0)
    relerr = torch.abs(potential_close_to_zero - exact_value) / exact_value
    assert relerr.item() < 3e-14


def test_exponent_out_of_range():
    match = r"`exponent` p=.* has to satisfy 0 < p <= 3"
    with pytest.raises(ValueError, match=match):
        InversePowerLawPotential(exponent=-1.0, smearing=0.0)

    with pytest.raises(ValueError, match=match):
        InversePowerLawPotential(exponent=4, smearing=0.0)


@pytest.mark.parametrize("smearing", smearings)
def test_inverserp_coulomb(smearing):
    """Check that an explicit Coulomb potential
    matches the 1/r^p implementation with p=1."""

    # Compute LR part of Coulomb potential using the potentials class working for any
    # exponent
    ipl = InversePowerLawPotential(exponent=1.0, smearing=smearing)
    coul = CoulombPotential(smearing=smearing)

    ipl_from_dist = ipl.from_dist(dists)
    ipl_from_dist_sq = ipl.from_dist_sq(dists_sq)
    ipl_sr_from_dist = ipl.sr_from_dist(dists)
    ipl_lr_from_dist = ipl.lr_from_dist(dists_sq)
    ipl_fourier = ipl.from_k_sq(ks_sq)

    coul_from_dist = coul.from_dist(dists)
    coul_from_dist_sq = coul.from_dist_sq(dists_sq)
    coul_sr_from_dist = coul.sr_from_dist(dists)
    coul_lr_from_dist = coul.lr_from_dist(dists_sq)
    coul_fourier = coul.from_k_sq(ks_sq)

    # Test agreement between generic and specialized implementations
    atol = 3e-16
    rtol = 2 * machine_epsilon
    assert_close(ipl_from_dist, coul_from_dist, rtol=rtol, atol=atol)
    assert_close(ipl_from_dist_sq, coul_from_dist_sq, rtol=rtol, atol=atol)

    atol = 3e-8
    rtol = 2 * machine_epsilon
    assert_close(ipl_sr_from_dist, coul_sr_from_dist, rtol=rtol, atol=atol)
    assert_close(ipl_lr_from_dist, coul_lr_from_dist, rtol=rtol, atol=atol)
    assert_close(ipl_fourier, coul_fourier, rtol=rtol, atol=atol)
