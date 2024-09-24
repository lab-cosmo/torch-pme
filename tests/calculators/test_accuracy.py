import pytest
import torch
from test_values_ewald import define_crystal
from utils import neighbor_list_torch

from torchpme import EwaldPotential


@pytest.mark.parametrize("calc_name", ["ewald"])
@pytest.mark.parametrize("rtol", [1e-3, 1e-6])
def test_parameter_choose(rtol, calc_name):
    """
    Check that the Madelung constants obtained from the Ewald sum calculator matches
    the reference values.
    In this test, only the charge-neutral crystal systems are chosen for which the
    potential converges relatively quickly, while the systems with a net charge are
    treated separately below.
    The structures cover a broad range of simple crystals, with cells ranging from cubic
    to triclinic, as well as cation-anion ratios of 1:1, 1:2 and 2:1.
    """
    # Get input parameters and adjust to account for scaling
    pos, charges, cell, madelung_ref, num_units = define_crystal()
    charges = charges.reshape((-1, 1))

    # Define calculator and tolerances
    if calc_name == "ewald":
        calc, sr_cutoff = EwaldPotential.from_accuracy(None, rtol, pos, charges, cell)
    elif calc_name == "pme":
        pass

    # Compute neighbor list
    neighbor_indices, neighbor_distances = neighbor_list_torch(
        positions=pos, periodic=True, box=cell, cutoff=sr_cutoff
    )

    # Compute potential and compare against target value using default hypers
    potentials = calc.forward(
        positions=pos,
        charges=charges,
        cell=cell,
        neighbor_indices=neighbor_indices,
        neighbor_distances=neighbor_distances,
    )
    energies = potentials * charges
    madelung = -torch.sum(energies) / num_units

    torch.testing.assert_close(madelung, madelung_ref, atol=0, rtol=rtol)
