import torch

from ..calculators import get_cscl_data as get_cscl_data_torch
from .base import Calculator
from .ewald import EwaldCalculator
from .pme import PMECalculator

__all__ = ["Calculator", "PMECalculator", "EwaldCalculator"]


def get_cscl_data():
    """
    Returns a CsCl structure (with a lattice parameter of 1 unit)
    and the neighbor list data.
    """

    from metatensor.torch import Labels, TensorBlock
    from metatensor.torch.atomistic import System

    # Get CsCl data in pure torch tensors
    charges, cell, positions, neighbor_distances, neighbor_indices = (
        get_cscl_data_torch()
    )
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

    # WIP, must still add neighbor list!

    return system
