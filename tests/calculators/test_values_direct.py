import math

import pytest
import torch

from torchpme import DirectPotential


def define_molecule(molecule_name="dimer"):
    """
    Define simple "molecules" (collection of point charges) for which the exact Coulomb
    potential is easy to evaluate. The implementations in the main code are then tested
    against these structures.
    """
    # Use a higher precision than the default float32
    dtype = torch.float64
    SQRT2 = torch.sqrt(torch.tensor(2.0, dtype=dtype))
    SQRT3 = torch.sqrt(torch.tensor(3.0, dtype=dtype))

    # Start defining molecules
    # Dimer
    if molecule_name == "dimer":
        positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=dtype)
        charges = torch.tensor([1.0, -1.0], dtype=dtype)
        potentials = torch.tensor([-1.0, 1.0], dtype=dtype)

    elif molecule_name == "dimer_positive":
        positions, charges, potentials = define_molecule("dimer")
        charges = torch.tensor([1.0, 1.0], dtype=dtype)
        potentials = torch.tensor([1.0, 1.0], dtype=dtype)

    elif molecule_name == "dimer_negative":
        positions, charges, potentials = define_molecule("dimer_positive")
        charges *= -1.0
        potentials *= -1.0

    # Equilateral triangle
    elif molecule_name == "triangle":
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1 / 2, SQRT3 / 2, 0.0]], dtype=dtype
        )
        charges = torch.tensor([1.0, -1.0, 0.0], dtype=dtype)
        potentials = torch.tensor([-1.0, 1.0, 0.0], dtype=dtype)

    elif molecule_name == "triangle_positive":
        positions, charges, potentials = define_molecule("triangle")
        charges = torch.tensor([1.0, 1.0, 1.0], dtype=dtype)
        potentials = torch.tensor([2.0, 2.0, 2.0], dtype=dtype)

    elif molecule_name == "triangle_negative":
        positions, charges, potentials = define_molecule("triangle_positive")
        charges *= -1.0
        potentials *= -1.0

    # Squares (planar)
    elif molecule_name == "square":
        positions = torch.tensor(
            [[1.0, 1.0, 0.0], [1.0, -1.0, 0.0], [-1.0, 1.0, 0.0], [-1.0, -1.0, 0.0]],
            dtype=dtype,
        )
        positions /= 2.0
        charges = torch.tensor([1.0, -1.0, -1.0, 1.0], dtype=dtype)
        potentials = charges * (1.0 / SQRT2 - 2.0)

    elif molecule_name == "square_positive":
        positions, charges, potentials = define_molecule("square")
        charges = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=dtype)
        potentials = (2.0 + 1.0 / SQRT2) * torch.ones(4, dtype=dtype)

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
            dtype=dtype,
        )
        charges = torch.tensor([1.0, -1.0, 1.0, -1.0], dtype=dtype)
        potentials = -charges

    elif molecule_name == "tetrahedron_positive":
        positions, charges, potentials = define_molecule("tetrahedron")
        charges = torch.ones(4, dtype=dtype)
        potentials = 3 * torch.ones(4, dtype=dtype)

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
    dtype = torch.float64

    # first rotation matrix: identity
    rot_1 = torch.eye(3, dtype=dtype)

    # second rotation matrix: rotation by angle phi around z-axis
    phi = 0.82321
    rot_2 = torch.zeros((3, 3), dtype=dtype)
    rot_2[0, 0] = rot_2[1, 1] = math.cos(phi)
    rot_2[0, 1] = -math.sin(phi)
    rot_2[1, 0] = math.sin(phi)
    rot_2[2, 2] = 1.0

    # third rotation matrix: second matrix followed by rotation by angle theta around y
    theta = 1.23456
    rot_3 = torch.zeros((3, 3), dtype=dtype)
    rot_3[0, 0] = rot_3[2, 2] = math.cos(theta)
    rot_3[0, 2] = math.sin(theta)
    rot_3[2, 0] = -math.sin(theta)
    rot_3[1, 1] = 1.0
    rot_3 = rot_3 @ rot_2

    # add additional orthogonal transformations by combining inversion
    transformations = [rot_1, rot_2, rot_3, -rot_1, -rot_3]

    for q in transformations:
        id = torch.eye(3, dtype=dtype)
        id_2 = q.T @ q
        torch.testing.assert_close(id, id_2, atol=2e-15, rtol=1e-14)
    return transformations


molecules = ["dimer", "triangle", "square", "tetrahedron"]
molecule_charges = ["", "_positive", "_negative"]
scaling_factors = torch.tensor([0.079, 1.0, 5.54], dtype=torch.float64)
orthogonal_transformations = generate_orthogonal_transformations()


@pytest.mark.parametrize("molecule", molecules)
@pytest.mark.parametrize("molecule_charge", molecule_charges)
@pytest.mark.parametrize("scaling_factor", scaling_factors)
@pytest.mark.parametrize("orthogonal_transformation", orthogonal_transformations)
def test_coulomb_exact(
    molecule, molecule_charge, scaling_factor, orthogonal_transformation
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
    direct = DirectPotential()

    # Compute potential at the position of the atoms for the specified structure
    molecule_name = molecule + molecule_charge
    positions, charges, ref_potentials = define_molecule(molecule_name)
    positions = scaling_factor * (positions @ orthogonal_transformation)
    potentials = direct.compute(positions, charges=charges)
    ref_potentials /= scaling_factor

    torch.testing.assert_close(potentials, ref_potentials, atol=2e-15, rtol=1e-14)


def test_direct_neighbor_indices():
    """test triangle with selected indices same as dimer"""

    direct = DirectPotential()

    positions, charges, _ = define_molecule(molecule_name="triangle")
    _, _, ref_potentials = define_molecule(molecule_name="dimer")

    neighbor_indices = torch.tensor([[0, 1], [1, 0]])

    potentials = direct.compute(
        positions=positions, charges=charges, neighbor_indices=neighbor_indices
    )

    assert torch.equal(potentials[:2], ref_potentials)
    assert potentials[2] == 0


def test_direct_cell():
    """Test a dimer which is "cut" accross a box.

    +----------+
    |          |
    | +      - |
    |          |
    +----------+
    """

    direct = DirectPotential()

    positions = torch.tensor([[0.5, 2.5, 0], [4.5, 2.5, 0]])
    charges = torch.tensor([1.0, -1.0]).unsqueeze(1)
    cell = 5.0 * torch.eye(3)
    neighbor_shifts = torch.tensor([[1, 0, 0], [-1, 0, 0]])

    # Test without PBC
    potentials_vacuum = direct.compute(positions=positions, charges=charges)
    ref_potentials_vacuum = torch.tensor([-1 / 4, 1 / 4]).unsqueeze(1)

    assert torch.equal(potentials_vacuum, ref_potentials_vacuum)

    # Test with PBC
    potentials_pbc = direct.compute(
        positions=positions, charges=charges, cell=cell, neighbor_shifts=neighbor_shifts
    )
    ref_potentials_pbc = torch.tensor([-1, 1]).unsqueeze(1)

    assert torch.equal(potentials_pbc, ref_potentials_pbc)


def test_direct_neighbor_indices_and_cell():
    """Test a selected dimer from a dimer across a box.

    +----------+
    |          |
    | +      - |
    | 0        |
    +----------+
    """

    direct = DirectPotential()

    positions = torch.tensor([[0.5, 2.5, 0], [4.5, 2.5, 0], [0.5, 0.0, 0.0]])
    charges = torch.tensor([1.0, -1.0, 0.0]).unsqueeze(1)
    cell = 5.0 * torch.eye(3)

    neighbor_indices = torch.tensor([[1, 0], [0, 1]])
    neighbor_shifts = torch.tensor([[1, 0, 0], [-1, 0, 0]])

    potentials = direct.compute(
        positions=positions,
        charges=charges,
        cell=cell,
        neighbor_indices=neighbor_indices,
        neighbor_shifts=neighbor_shifts,
    )
    ref_potentials = torch.tensor([-1, 1, 0]).unsqueeze(1)

    assert torch.equal(potentials, ref_potentials)
