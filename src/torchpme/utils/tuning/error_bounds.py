import math
import torch

from . import TuningErrorBounds

TWO_PI = 2 * math.pi

class EwaldErrorBounds(TuningErrorBounds):
    r"""
    Error bounds for :class:`torchpme.calculators.ewald.EwaldCalculator`.

    The error formulas are given `online
    <https://www2.icp.uni-stuttgart.de/~icp/mediawiki/images/4/4d/Script_Longrange_Interactions.pdf>`_
    (now not available, need to be updated later). Note the difference notation between
    the parameters in the reference and ours:

    .. math::

        \alpha &= \left( \sqrt{2}\,\mathrm{smearing} \right)^{-1}

        K &= \frac{2 \pi}{\mathrm{lr\_wavelength}}

        r_c &= \mathrm{cutoff}

    :param charges: atomic charges
    :param cell: single tensor of shape (3, 3), describing the bounding
    :param positions: single tensor of shape (``len(charges), 3``) containing the
        Cartesian positions of all point charges in the system.
    """

    def __init__(
        self,
        charges: torch.Tensor,
        cell: torch.Tensor,
        positions: torch.Tensor,
    ):
        super().__init__(charges, cell, positions)

        self.volume = torch.abs(torch.det(cell))
        self.sum_squared_charges = (charges**2).sum()
        self.prefac = 2 * self.sum_squared_charges / math.sqrt(len(positions))
        self.cell = cell
        self.positions = positions

    def err_kspace(self, smearing, lr_wavelength):
        return (
            self.prefac**0.5
            / smearing
            / torch.sqrt(TWO_PI**2 * self.volume / (lr_wavelength) ** 0.5)
            * torch.exp(-(TWO_PI**2) * smearing**2 / (lr_wavelength))
        )

    def err_rspace(self, smearing, cutoff):
        return (
            self.prefac
            / torch.sqrt(cutoff * self.volume)
            * torch.exp(-(cutoff**2) / 2 / smearing**2)
        )

    def forward(self, smearing, lr_wavelength, cutoff):
        r"""
        Calculate the error bound of Ewald.

        :param smearing: see :class:`torchpme.EwaldCalculator` for details
        :param lr_wavelength: see :class:`torchpme.EwaldCalculator` for details
        :param cutoff: see :class:`torchpme.EwaldCalculator` for details
        """
        smearing = torch.as_tensor(smearing)
        lr_wavelength = torch.as_tensor(lr_wavelength)
        cutoff = torch.as_tensor(cutoff)
        return torch.sqrt(
            self.err_kspace(smearing, lr_wavelength) ** 2
            + self.err_rspace(smearing, cutoff) ** 2
        )
    

# Coefficients for the P3M Fourier error,
# see Table II of http://dx.doi.org/10.1063/1.477415
A_COEF = [
    [None, 2 / 3, 1 / 50, 1 / 588, 1 / 4320, 1 / 23_232, 691 / 68_140_800, 1 / 345_600],
    [
        None,
        None,
        5 / 294,
        7 / 1440,
        3 / 1936,
        7601 / 13_628_160,
        13 / 57_600,
        3617 / 35_512_320,
    ],
    [
        None,
        None,
        None,
        21 / 3872,
        7601 / 2_271_360,
        143 / 69_120,
        47_021 / 35_512_320,
        745_739 / 838_397_952,
    ],
    [
        None,
        None,
        None,
        None,
        143 / 28_800,
        517_231 / 106_536_960,
        9_694_607 / 2_095_994_880,
        56_399_353 / 12_773_376_000,
    ],
    [
        None,
        None,
        None,
        None,
        None,
        106_640_677 / 11_737_571_328,
        733_191_589 / 59_609_088_000,
        25_091_609 / 1_560_084_480,
    ],
    [
        None,
        None,
        None,
        None,
        None,
        None,
        326_190_917 / 11_700_633_600,
        1_755_948_832_039 / 36_229_939_200_000,
    ],
    [None, None, None, None, None, None, None, 4_887_769_399 / 37_838_389_248],
]


class P3MErrorBounds(TuningErrorBounds):
    r"""
    "
    Error bounds for :class:`torchpme.calculators.pme.P3MCalculator`.

    For the error formulas are given `here <https://doi.org/10.1063/1.477415>`_.
    Note the difference notation between the parameters in the reference and ours:

    .. math::

        \alpha = \left(\sqrt{2}\,\mathrm{smearing} \right)^{-1}

    :param charges: atomic charges
    :param cell: single tensor of shape (3, 3), describing the bounding
    :param positions: single tensor of shape (``len(charges), 3``) containing the
        Cartesian positions of all point charges in the system.
    """

    def __init__(
        self, charges: torch.Tensor, cell: torch.Tensor, positions: torch.Tensor
    ):
        super().__init__(charges, cell, positions)

        self.volume = torch.abs(torch.det(cell))
        self.sum_squared_charges = (charges**2).sum()
        self.prefac = 2 * self.sum_squared_charges / math.sqrt(len(positions))
        self.cell_dimensions = torch.linalg.norm(cell, dim=1)
        self.cell = cell
        self.positions = positions

    def err_kspace(self, smearing, mesh_spacing, interpolation_nodes):
        actual_spacing = self.cell_dimensions / (
            2 * self.cell_dimensions / mesh_spacing + 1
        )
        h = torch.prod(actual_spacing) ** (1 / 3)

        return (
            self.prefac
            / self.volume ** (2 / 3)
            * (h * (1 / 2**0.5 / smearing)) ** interpolation_nodes
            * torch.sqrt(
                (1 / 2**0.5 / smearing)
                * self.volume ** (1 / 3)
                * math.sqrt(2 * torch.pi)
                * sum(
                    A_COEF[m][interpolation_nodes]
                    * (h * (1 / 2**0.5 / smearing)) ** (2 * m)
                    for m in range(interpolation_nodes)
                )
            )
        )

    def err_rspace(self, smearing, cutoff):
        return (
            self.prefac
            / torch.sqrt(cutoff * self.volume)
            * torch.exp(-(cutoff**2) / 2 / smearing**2)
        )

    def forward(self, smearing, mesh_spacing, cutoff, interpolation_nodes):
        r"""
        Calculate the error bound of P3M.

        :param smearing: see :class:`torchpme.P3MCalculator` for details
        :param mesh_spacing: see :class:`torchpme.P3MCalculator` for details
        :param cutoff: see :class:`torchpme.P3MCalculator` for details
        :param interpolation_nodes: The number ``n`` of nodes used in the interpolation
            per coordinate axis. The total number of interpolation nodes in 3D will be
            ``n^3``. In general, for ``n`` nodes, the interpolation will be performed by
            piecewise polynomials of degree ``n`` (e.g. ``n = 3`` for cubic
            interpolation). Only the values ``1, 2, 3, 4, 5`` are supported.
        """
        smearing = torch.as_tensor(smearing)
        mesh_spacing = torch.as_tensor(mesh_spacing)
        cutoff = torch.as_tensor(cutoff)
        interpolation_nodes = torch.as_tensor(interpolation_nodes)
        return torch.sqrt(
            self.err_kspace(smearing, mesh_spacing, interpolation_nodes) ** 2
            + self.err_rspace(smearing, cutoff) ** 2
        )
    

class PMEErrorBounds(TuningErrorBounds):
    r"""
    Error bounds for :class:`torchpme.PMECalculator`.
    For the error formulas are given `elsewhere <https://doi.org/10.1063/1.470043>`_.
    Note the difference notation between the parameters in the reference and ours:

    .. math::

        \alpha = \left(\sqrt{2}\,\mathrm{smearing} \right)^{-1}

    :param charges: atomic charges
    :param cell: single tensor of shape (3, 3), describing the bounding
    :param positions: single tensor of shape (``len(charges), 3``) containing the
        Cartesian positions of all point charges in the system.
    """

    def __init__(
        self, charges: torch.Tensor, cell: torch.Tensor, positions: torch.Tensor
    ):
        super().__init__(charges, cell, positions)

        self.volume = torch.abs(torch.det(cell))
        self.sum_squared_charges = (charges**2).sum()
        self.prefac = 2 * self.sum_squared_charges / math.sqrt(len(positions))
        self.cell_dimensions = torch.linalg.norm(cell, dim=1)

    def err_kspace(self, smearing, mesh_spacing, interpolation_nodes):
        actual_spacing = self.cell_dimensions / (
            2 * self.cell_dimensions / mesh_spacing + 1
        )
        h = torch.prod(actual_spacing) ** (1 / 3)
        i_n_factorial = torch.exp(torch.lgamma(interpolation_nodes + 1))
        RMS_phi = [None, None, 0.246, 0.404, 0.950, 2.51, 8.42]

        return (
            self.prefac
            * torch.pi**0.25
            * (6 * (1 / 2**0.5 / smearing) / (2 * interpolation_nodes + 1)) ** 0.5
            / self.volume ** (2 / 3)
            * (2**0.5 / smearing * h) ** interpolation_nodes
            / i_n_factorial
            * torch.exp(
                interpolation_nodes * (torch.log(interpolation_nodes / 2) - 1) / 2
            )
            * RMS_phi[interpolation_nodes - 1]
        )

    def err_rspace(self, smearing, cutoff):
        smearing = torch.as_tensor(smearing)
        cutoff = torch.as_tensor(cutoff)

        return (
            self.prefac
            / torch.sqrt(cutoff * self.volume)
            * torch.exp(-(cutoff**2) / 2 / smearing**2)
        )

    def error(self, cutoff, smearing, mesh_spacing, interpolation_nodes):
        r"""
        Calculate the error bound of PME.

        :param smearing: if its value is given, it will not be tuned, see
            :class:`torchpme.PMECalculator` for details
        :param mesh_spacing: if its value is given, it will not be tuned, see
            :class:`torchpme.PMECalculator` for details
        :param cutoff: if its value is given, it will not be tuned, see
            :class:`torchpme.PMECalculator` for details
        :param interpolation_nodes: The number ``n`` of nodes used in the interpolation
            per coordinate axis. The total number of interpolation nodes in 3D will be
            ``n^3``. In general, for ``n`` nodes, the interpolation will be performed by
            piecewise polynomials of degree ``n - 1`` (e.g. ``n = 4`` for cubic
            interpolation). Only the values ``3, 4, 5, 6, 7`` are supported.
        """
        smearing = torch.as_tensor(smearing)
        mesh_spacing = torch.as_tensor(mesh_spacing)
        cutoff = torch.as_tensor(cutoff)
        interpolation_nodes = torch.as_tensor(interpolation_nodes)
        return torch.sqrt(
            self.err_rspace(smearing, cutoff) ** 2
            + self.err_kspace(smearing, mesh_spacing, interpolation_nodes) ** 2
        )

