import math

import pytest
import torch
from utils import neighbor_list_torch

from torchpme import DirectPotential

DTYPE = torch.float64


def define_molecule(molecule_name="dimer"):
    """
    Define simple "molecules" (collection of point charges) for which the exact Coulomb
    potential is easy to evaluate. The implementations in the main code are then tested
    against these structures.
    """
    SQRT2 = torch.sqrt(torch.tensor(2.0, dtype=DTYPE))
    SQRT3 = torch.sqrt(torch.tensor(3.0, dtype=DTYPE))

    # Start defining molecules
    # Dimer
    if molecule_name == "dimer":
        positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=DTYPE)
        charges = torch.tensor([1.0, -1.0], dtype=DTYPE)
        potentials = torch.tensor([-1.0, 1.0], dtype=DTYPE)

    elif molecule_name == "dimer_positive":
        positions, charges, potentials = define_molecule("dimer")
        charges = torch.tensor([1.0, 1.0], dtype=DTYPE)
        potentials = torch.tensor([1.0, 1.0], dtype=DTYPE)

    elif molecule_name == "dimer_negative":
        positions, charges, potentials = define_molecule("dimer_positive")
        charges *= -1.0
        potentials *= -1.0

    # Equilateral triangle
    elif molecule_name == "triangle":
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1 / 2, SQRT3 / 2, 0.0]], dtype=DTYPE
        )
        charges = torch.tensor([1.0, -1.0, 0.0], dtype=DTYPE)
        potentials = torch.tensor([-1.0, 1.0, 0.0], dtype=DTYPE)

    elif molecule_name == "triangle_positive":
        positions, charges, potentials = define_molecule("triangle")
        charges = torch.tensor([1.0, 1.0, 1.0], dtype=DTYPE)
        potentials = torch.tensor([2.0, 2.0, 2.0], dtype=DTYPE)

    elif molecule_name == "triangle_negative":
        positions, charges, potentials = define_molecule("triangle_positive")
        charges *= -1.0
        potentials *= -1.0

    # Squares (planar)
    elif molecule_name == "square":
        positions = torch.tensor(
            [[1.0, 1.0, 0.0], [1.0, -1.0, 0.0], [-1.0, 1.0, 0.0], [-1.0, -1.0, 0.0]],
            dtype=DTYPE,
        )
        positions /= 2.0
        charges = torch.tensor([1.0, -1.0, -1.0, 1.0], dtype=DTYPE)
        potentials = charges * (1.0 / SQRT2 - 2.0)

    elif molecule_name == "square_positive":
        positions, charges, potentials = define_molecule("square")
        charges = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=DTYPE)
        potentials = (2.0 + 1.0 / SQRT2) * torch.ones(4, dtype=DTYPE)

    elif molecule_name == "square_negative":
        positions, charges, potentials = define_molecule("square_positive")
        charges *= -1.0
        potentials *= -1.0

    # Tetrahedra
    elif molecule_name == "tetrahedron":
        positions = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1 / 2, SQRT3 / 2, 0.0],
                [1 / 2, SQRT3 / 6, SQRT2 / SQRT3],
            ],
            dtype=DTYPE,
        )
        charges = torch.tensor([1.0, -1.0, 1.0, -1.0], dtype=DTYPE)
        potentials = -charges

    elif molecule_name == "tetrahedron_positive":
        positions, charges, potentials = define_molecule("tetrahedron")
        charges = torch.ones(4, dtype=DTYPE)
        potentials = 3 * torch.ones(4, dtype=DTYPE)

    elif molecule_name == "tetrahedron_negative":
        positions, charges, potentials = define_molecule("tetrahedron_positive")
        charges *= -1.0
        potentials *= -1.0
    else:
        raise ValueError(f"unknown molecule_name {molecule_name}")

    charges = charges.reshape((-1, 1))
    potentials = potentials.reshape((-1, 1))

    return positions, charges, potentials


def generate_orthogonal_transformations():
    # first rotation matrix: identity
    rot_1 = torch.eye(3, dtype=DTYPE)

    # second rotation matrix: rotation by angle phi around z-axis
    phi = 0.82321
    rot_2 = torch.zeros((3, 3), dtype=DTYPE)
    rot_2[0, 0] = rot_2[1, 1] = math.cos(phi)
    rot_2[0, 1] = -math.sin(phi)
    rot_2[1, 0] = math.sin(phi)
    rot_2[2, 2] = 1.0

    # third rotation matrix: second matrix followed by rotation by angle theta around y
    theta = 1.23456
    rot_3 = torch.zeros((3, 3), dtype=DTYPE)
    rot_3[0, 0] = rot_3[2, 2] = math.cos(theta)
    rot_3[0, 2] = math.sin(theta)
    rot_3[2, 0] = -math.sin(theta)
    rot_3[1, 1] = 1.0
    rot_3 = rot_3 @ rot_2

    # add additional orthogonal transformations by combining inversion
    transformations = [rot_1, rot_2, rot_3, -rot_1, -rot_3]

    for q in transformations:
        id = torch.eye(3, dtype=DTYPE)
        id_2 = q.T @ q
        torch.testing.assert_close(id, id_2, atol=2e-15, rtol=1e-14)
    return transformations


molecules = ["dimer", "triangle", "square", "tetrahedron"]
molecule_charges = ["", "_positive", "_negative"]
scaling_factors = [0.079, 1.0, 5.54]
orthogonal_transformations = generate_orthogonal_transformations()
full_neighbor_list = [True, False]


@pytest.mark.parametrize("molecule", molecules)
@pytest.mark.parametrize("molecule_charge", molecule_charges)
@pytest.mark.parametrize("scaling_factor", scaling_factors)
@pytest.mark.parametrize("orthogonal_transformation", orthogonal_transformations)
@pytest.mark.parametrize("full_neighbor_list", full_neighbor_list)
def test_coulomb_exact(
    molecule,
    molecule_charge,
    scaling_factor,
    orthogonal_transformation,
    full_neighbor_list,
):
    """
    Check that the Coulomb potentials obtained from the calculators match the correct
    value for simple toy systems.
    To make the test stricter, the molecules are also rotated and scaled by varying
    amounts, the former of which leaving the potentials invariant, while the second
    operation scales the potentials by the inverse amount.
    """
    # Call Ewald potential class without specifying any of the convergence parameters
    # so that they are chosen by default (in a structure-dependent way)
    direct = DirectPotential(full_neighbor_list=full_neighbor_list)

    # Compute potential at the position of the atoms for the specified structure
    molecule_name = molecule + molecule_charge
    positions, charges, ref_potentials = define_molecule(molecule_name)
    positions = scaling_factor * (positions @ orthogonal_transformation)

    # Choose a large cutoff that covers all atoms
    neighbor_indices, neighbor_distances = neighbor_list_torch(
        positions=positions,
        periodic=False,
        cutoff=scaling_factor * 10,
        full_neighbor_list=full_neighbor_list,
    )

    potentials = direct.forward(
        positions,
        charges=charges,
        cell=torch.eye(3, dtype=DTYPE),  # ignored in actual calculations
        neighbor_indices=neighbor_indices,
        neighbor_distances=neighbor_distances,
    )
    # Divide by 2 due to double counting of atom pairs
    ref_potentials /= 2 * scaling_factor

    torch.testing.assert_close(potentials, ref_potentials, atol=2e-15, rtol=1e-14)
