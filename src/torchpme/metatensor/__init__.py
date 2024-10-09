import torch

from ..calculators import get_cscl_data as get_cscl_data_torch
from .base import Calculator
from .ewald import EwaldCalculator, tune_ewald
from .pme import PMECalculator, tune_pme

__all__ = [
    "Calculator",
    "PMECalculator",
    "EwaldCalculator",
    "tune_ewald",
    "tune_pme",
    "get_cscl_data",
]


def get_cscl_data():
    """
    Returns a CsCl structure (with a lattice parameter of 1 unit)
    and the neighbor list data, in `metatensor` format.
    """

    from metatensor.torch import Labels, TensorBlock
    from metatensor.torch.atomistic import System

    # Get CsCl data in pure torch tensors
    (
        charges,
        cell,
        positions,
        neighbor_distances,
        neighbor_indices,
        cell_shifts,
        displacements,
    ) = get_cscl_data_torch()
    types = torch.tensor([55, 17])

    # Convert the geometry in a `System` object
    system = System(
        types=types,
        positions=positions,
        cell=cell,
    ).to(dtype=torch.float64)

    # Charges can be included as an additional data block
    data = TensorBlock(
        values=charges,
        samples=Labels.range("atom", charges.shape[0]),
        components=[],
        properties=Labels.range("charge", charges.shape[1]),
    )
    system.add_data(name="charges", data=data)

    # `metatensor`-style neighbor lists contain also information that is
    # not strictly necessary for `torch-pme` (cell shifts, and displacement vectors)
    # that must combined in a  :py:class:`TensorBlock <metatensor.torch.TensorBlock>`
    # object.

    sample_values = torch.hstack([neighbor_indices, cell_shifts])
    samples = Labels(
        names=[
            "first_atom",
            "second_atom",
            "cell_shift_a",
            "cell_shift_b",
            "cell_shift_c",
        ],
        values=sample_values,
    )

    values = displacements.reshape(-1, 3, 1)
    neighbors = TensorBlock(
        values=values,
        samples=samples,
        components=[Labels.range("xyz", 3)],
        properties=Labels.range("distance", 1),
    )

    # NB: for this specific toy system, the neighbor-list cutoff is shorter than
    # the first-neighbor distance, s neighbor list the TensorBlock.
    # is empty.

    return system, neighbors
