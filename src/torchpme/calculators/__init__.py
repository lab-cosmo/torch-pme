import torch

from .base import Calculator, estimate_smearing
from .ewald import EwaldCalculator, tune_ewald
from .pme import PMECalculator

__all__ = [
    "Calculator",
    "EwaldCalculator",
    "PMECalculator",
    "estimate_smearing",
    "tune_ewald",
]


def get_cscl_data():
    """
    Returns a CsCl structure (with a lattice parameter of 1 unit)
    and the neighbor list data.
    """

    try:
        from vesin.torch import NeighborList
    except ImportError:
        print("We need the ``vesin`` package to compute the neighbor list.")
        raise

    # Define crystal structure
    charges = torch.tensor([[1.0], [-1.0]], dtype=torch.float64)
    cell = torch.eye(3, dtype=torch.float64)
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], dtype=torch.float64)

    # Compute the neighbor indices (``"i"``, ``"j"``) and the neighbor
    # distances ("``d``") using the ``vesin`` package. Refer to the
    # `documentation <https://luthaf.fr/vesin>`_ for details on the API.
    cell_dimensions = torch.linalg.norm(cell, dim=1)
    cutoff = torch.min(cell_dimensions) / 2 - 1e-6
    nl = NeighborList(cutoff=cutoff, full_list=False)
    i, j, neighbor_distances = nl.compute(
        points=positions, box=cell, periodic=True, quantities="ijd"
    )
    neighbor_indices = torch.stack([i, j], dim=1)

    return charges, cell, positions, neighbor_distances, neighbor_indices
