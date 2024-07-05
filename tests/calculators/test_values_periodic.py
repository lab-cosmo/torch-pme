import math
import os

import numpy as np
import pytest
import torch

# Imports for random structure
from ase.io import read

from meshlode import EwaldPotential, MeshEwaldPotential


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
    transformations = [rot_2, rot_3]

    # make sure that the generated transformations are indeed orthogonal
    for q in transformations:
        id = torch.eye(3, dtype=dtype)
        id_2 = q.T @ q
        torch.testing.assert_close(id, id_2, atol=2e-15, rtol=1e-14)
    return transformations


dtype = torch.float64


def define_crystal(crystal_name="CsCl"):
    # Define all relevant parameters (atom positions, charges, cell) of the reference
    # crystal structures for which the Madelung constants obtained from the Ewald sums
    # are compared with reference values.
    # see https://www.sciencedirect.com/science/article/pii/B9780128143698000078#s0015
    # More detailed values can be found in https://pubs.acs.org/doi/10.1021/ic2023852

    # Caesium-Chloride (CsCl) structure:
    # - Cubic unit cell
    # - 1 atom pair in the unit cell
    # - Cation-Anion ratio of 1:1
    if crystal_name == "CsCl":
        types = torch.tensor([17, 55])  # Cl and Cs
        positions = torch.tensor([[0, 0, 0], [0.5, 0.5, 0.5]], dtype=dtype)
        charges = torch.tensor([-1.0, 1.0], dtype=dtype)
        cell = torch.eye(3, dtype=dtype)
        madelung_ref = 2.035361
        num_formula_units = 1

    # Sodium-Chloride (NaCl) structure using a primitive unit cell
    # - non-cubic unit cell (fcc)
    # - 1 atom pair in the unit cell
    # - Cation-Anion ratio of 1:1
    elif crystal_name == "NaCl_primitive":
        types = torch.tensor([11, 17])
        positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=dtype)
        charges = torch.tensor([1.0, -1.0], dtype=dtype)
        cell = torch.tensor([[0, 1.0, 1], [1, 0, 1], [1, 1, 0]], dtype=dtype)  # fcc
        madelung_ref = 1.74756
        num_formula_units = 1

    # Sodium-Chloride (NaCl) structure using a cubic unit cell
    # - cubic unit cell
    # - 4 atom pairs in the unit cell
    # - Cation-Anion ratio of 1:1
    elif crystal_name == "NaCl_cubic":
        types = torch.tensor([11, 17, 17, 17, 11, 11, 11, 17])
        positions = torch.tensor(
            [
                [0.0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 1, 0],
                [1, 0, 1],
                [0, 1, 1],
                [1, 1, 1],
            ],
            dtype=dtype,
        )
        charges = torch.tensor([+1.0, -1, -1, -1, +1, +1, +1, -1], dtype=dtype)
        cell = 2 * torch.eye(3, dtype=dtype)
        madelung_ref = 1.747565
        num_formula_units = 4

    # ZnS (zincblende) structure
    # - non-cubic unit cell (fcc)
    # - 1 atom pair in the unit cell
    # - Cation-Anion ratio of 1:1
    # Remarks: we use a primitive unit cell which makes the lattice parameter of the
    # cubic cell equal to 2.
    elif crystal_name == "zincblende":
        types = torch.tensor([16, 30])
        positions = torch.tensor([[0, 0, 0], [0.5, 0.5, 0.5]], dtype=dtype)
        charges = torch.tensor([1.0, -1], dtype=dtype)
        cell = torch.tensor([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=dtype)
        madelung_ref = 2 * 1.63806 / np.sqrt(3)
        num_formula_units = 1

    # Wurtzite structure
    # - non-cubic unit cell (triclinic)
    # - 2 atom pairs in the unit cell
    # - Cation-Anion ratio of 1:1
    elif crystal_name == "wurtzite":
        u = 3 / 8
        c = np.sqrt(1 / u)
        types = torch.tensor([16, 30, 16, 30])
        positions = torch.tensor(
            [
                [0.5, 0.5 / np.sqrt(3), 0.0],
                [0.5, 0.5 / np.sqrt(3), u * c],
                [0.5, -0.5 / np.sqrt(3), 0.5 * c],
                [0.5, -0.5 / np.sqrt(3), (0.5 + u) * c],
            ],
            dtype=dtype,
        )
        charges = torch.tensor([1.0, -1, 1, -1], dtype=dtype)
        cell = torch.tensor(
            [[0.5, -0.5 * np.sqrt(3), 0], [0.5, 0.5 * np.sqrt(3), 0], [0, 0, c]],
            dtype=dtype,
        )
        madelung_ref = 1.64132 / (u * c)
        num_formula_units = 2

    # Fluorite structure (e.g. CaF2 with Ca2+ and F-)
    # - non-cubic (fcc) unit cell
    # - 1 neutral molecule per unit cell
    # - Cation-Anion ratio of 1:2
    elif crystal_name == "fluorite":
        a = 5.463
        a = 1.0
        types = torch.tensor([9, 9, 20])
        positions = a * torch.tensor(
            [[1 / 4, 1 / 4, 1 / 4], [3 / 4, 3 / 4, 3 / 4], [0, 0, 0]], dtype=dtype
        )
        charges = torch.tensor([-1, -1, 2], dtype=dtype)
        cell = torch.tensor([[a, a, 0], [a, 0, a], [0, a, a]], dtype=dtype) / 2.0
        madelung_ref = 11.636575
        num_formula_units = 1

    # Copper(I)-Oxide structure (e.g. Cu2O with Cu+ and O2-)
    # - cubic unit cell
    # - 2 neutral molecules per unit cell
    # - Cation-Anion ratio of 2:1
    elif crystal_name == "cu2o":
        a = 1.0
        types = torch.tensor([8, 8, 29, 29, 29, 29])
        positions = a * torch.tensor(
            [
                [0, 0, 0],
                [1 / 2, 1 / 2, 1 / 2],
                [1 / 4, 1 / 4, 1 / 4],
                [1 / 4, 3 / 4, 3 / 4],
                [3 / 4, 1 / 4, 3 / 4],
                [3 / 4, 3 / 4, 1 / 4],
            ],
            dtype=dtype,
        )
        charges = torch.tensor([-2, -2, 1, 1, 1, 1], dtype=dtype)
        cell = a * torch.eye(3, dtype=dtype)
        madelung_ref = 10.2594570330750
        num_formula_units = 2

    # Wigner crystal in simple cubic structure.
    # Wigner crystals are equivalent to the Jellium or uniform electron gas models.
    # For the purpose of this test, we define them to be structures in which the ion
    # cores form a perfect lattice, while the electrons are uniformly distributed over
    # the cell. In some sources, the role of the positive and negative charges are
    # flipped. These structures are used to test the code for cases in which the total
    # charge of the particles is not zero.
    # Wigner crystal energies are taken from "Zero-Point Energy of an Electron Lattice"
    # by Rosemary A., Coldwell‐Horsfall and Alexei A. Maradudin (1960), eq. (A21).
    elif crystal_name == "wigner_sc":
        types = torch.tensor([1])
        positions = torch.tensor([[0, 0, 0]], dtype=dtype)
        charges = torch.tensor([1.0], dtype=dtype)
        cell = torch.tensor([[1.0, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=dtype)

        # Reference value is expressed in terms of the Wigner-Seiz radius, and needs to
        # be rescaled to the case in which the lattice parameter = 1.
        madelung_wigner_seiz = 1.7601188
        wigner_seiz_radius = (3 / (4 * np.pi)) ** (1 / 3)
        madelung_ref = madelung_wigner_seiz / wigner_seiz_radius  # 2.83730
        num_formula_units = 1

    # Wigner crystal in bcc structure (note: this is the most stable structure).
    # See description of "wigner_sc" for a general explanation on Wigner crystals.
    # Used to test the code for cases in which the unit cell has a nonzero net charge.
    elif crystal_name == "wigner_bcc":
        types = torch.tensor([1])
        positions = torch.tensor([[0, 0, 0]], dtype=dtype)
        charges = torch.tensor([1.0], dtype=dtype)
        cell = torch.tensor(
            [[1.0, 0, 0], [0, 1, 0], [1 / 2, 1 / 2, 1 / 2]], dtype=dtype
        )

        # Reference value is expressed in terms of the Wigner-Seiz radius, and needs to
        # be rescaled to the case in which the lattice parameter = 1.
        madelung_wigner_seiz = 1.791860
        wigner_seiz_radius = (3 / (4 * np.pi * 2)) ** (
            1 / 3
        )  # 2 atoms per cubic unit cell
        madelung_ref = madelung_wigner_seiz / wigner_seiz_radius  # 3.63924
        num_formula_units = 1

    # Same as above, but now using a cubic unit cell rather than the primitive bcc cell
    elif crystal_name == "wigner_bcc_cubiccell":
        types = torch.tensor([1, 1])
        positions = torch.tensor([[0, 0, 0], [1 / 2, 1 / 2, 1 / 2]], dtype=dtype)
        charges = torch.tensor([1.0, 1.0], dtype=dtype)
        cell = torch.tensor([[1.0, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=dtype)

        # Reference value is expressed in terms of the Wigner-Seiz radius, and needs to
        # be rescaled to the case in which the lattice parameter = 1.
        madelung_wigner_seiz = 1.791860
        wigner_seiz_radius = (3 / (4 * np.pi * 2)) ** (
            1 / 3
        )  # 2 atoms per cubic unit cell
        madelung_ref = madelung_wigner_seiz / wigner_seiz_radius  # 3.63924
        num_formula_units = 2

    # Wigner crystal in fcc structure
    # See description of "wigner_sc" for a general explanation on Wigner crystals.
    # Used to test the code for cases in which the unit cell has a nonzero net charge.
    elif crystal_name == "wigner_fcc":
        types = torch.tensor([1])
        positions = torch.tensor([[0, 0, 0]], dtype=dtype)
        charges = torch.tensor([1.0], dtype=dtype)
        cell = torch.tensor([[1, 0, 1], [0, 1, 1], [1, 1, 0]], dtype=dtype) / 2

        # Reference value is expressed in terms of the Wigner-Seiz radius, and needs to
        # be rescaled to the case in which the lattice parameter = 1.
        madelung_wigner_seiz = 1.791753
        wigner_seiz_radius = (3 / (4 * np.pi * 4)) ** (
            1 / 3
        )  # 4 atoms per cubic unit cell
        madelung_ref = madelung_wigner_seiz / wigner_seiz_radius  # 4.58488
        num_formula_units = 1

    # Same as above, but now using a cubic unit cell rather than the primitive fcc cell
    elif crystal_name == "wigner_fcc_cubiccell":
        types = torch.tensor([1, 1, 1, 1])
        positions = 0.5 * torch.tensor(
            [[0.0, 0, 0], [1, 0, 1], [1, 1, 0], [0, 1, 1]], dtype=dtype
        )
        charges = torch.tensor([1.0, 1, 1, 1], dtype=dtype)
        cell = torch.eye(3, dtype=dtype)

        # Reference value is expressed in terms of the Wigner-Seiz radius, and needs to
        # be rescaled to the case in which the lattice parameter = 1.
        madelung_wigner_seiz = 1.791753
        wigner_seiz_radius = (3 / (4 * np.pi * 4)) ** (
            1 / 3
        )  # 4 atoms per cubic unit cell
        madelung_ref = madelung_wigner_seiz / wigner_seiz_radius  # 4.58488
        num_formula_units = 4

    else:
        raise ValueError(f"crystal_name = {crystal_name} is not supported!")

    madelung_ref = torch.tensor(madelung_ref, dtype=dtype)
    return types, positions, charges, cell, madelung_ref, num_formula_units


scaling_factors = torch.tensor([1 / 2.0353610, 1.0, 3.4951291], dtype=torch.float64)
neutral_crystals = ["CsCl", "NaCl_primitive", "NaCl_cubic", "zincblende", "wurtzite"]
neutral_crystals += ["cu2o", "fluorite"]


@pytest.mark.parametrize("calc_name", ["ewald", "pme"])
@pytest.mark.parametrize("crystal_name", neutral_crystals)
@pytest.mark.parametrize("scaling_factor", scaling_factors)
def test_madelung(crystal_name, scaling_factor, calc_name):
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
    types, pos, charges, cell, madelung_ref, num_units = define_crystal(crystal_name)
    pos *= scaling_factor
    cell *= scaling_factor
    madelung_ref /= scaling_factor
    charges = charges.reshape((-1, 1))

    # Define calculator and tolerances
    if calc_name == "ewald":
        sr_cutoff = scaling_factor * torch.tensor(1.0, dtype=dtype)
        calc = EwaldPotential(sr_cutoff=sr_cutoff)
        rtol = 4e-6
    elif calc_name == "pme":
        sr_cutoff = scaling_factor * torch.tensor(2.0, dtype=dtype)
        calc = MeshEwaldPotential(sr_cutoff=sr_cutoff)
        rtol = 9e-4

    # Compute potential and compare against target value using default hypers
    potentials = calc.compute(types, positions=pos, cell=cell, charges=charges)
    energies = potentials * charges
    madelung = -torch.sum(energies) / 2 / num_units

    torch.testing.assert_close(madelung, madelung_ref, atol=0.0, rtol=rtol)


# Since structures without charge neutrality show slower convergence, these
# structures are tested separately.
wigner_crystals = [
    "wigner_sc",
    "wigner_fcc",
    "wigner_fcc_cubiccell",
    "wigner_bcc",
    "wigner_bcc_cubiccell",
]

scaling_factors = torch.tensor([0.4325, 1.0, 2.0353610], dtype=torch.float64)


@pytest.mark.parametrize("crystal_name", wigner_crystals)
@pytest.mark.parametrize("scaling_factor", scaling_factors)
def test_wigner(crystal_name, scaling_factor):
    """
    Check that the energy of a Wigner solid  obtained from the Ewald sum calculator
    matches the reference values.
    In this test, the Wigner solids are defined by placing arranging positively charged
    point particles on a bcc lattice, leading to a net charge of the unit cell if we
    only look at the ions. This charge is compensated by a homogeneous neutral back-
    ground charge of opposite sign (physically: completely delocalized electrons).

    The presence of a net charge (due to the particles but without background) leads
    to numerically slower convergence of the relevant sums.
    """
    # Get parameters defining atomic positions, cell and charges
    types, positions, charges, cell, madelung_ref, num = define_crystal(crystal_name)
    positions *= scaling_factor
    cell *= scaling_factor
    madelung_ref /= scaling_factor

    # Due to the slow convergence, we do not use the default values of the smearing,
    # but provide a range instead. The first value of 0.1 corresponds to what would be
    # chosen by default for the "wigner_sc" or "wigner_bcc_cubiccell" structure.
    smearings = torch.tensor([0.1, 0.06, 0.019], dtype=torch.float64)
    tolerances = torch.tensor([3e-2, 1e-2, 1e-3])
    for smearing, rtol in zip(smearings, tolerances):
        # Readjust smearing parameter to match nearest neighbor distance
        if crystal_name in ["wigner_fcc", "wigner_fcc_cubiccell"]:
            smeareff = smearing / np.sqrt(2)
        elif crystal_name in ["wigner_bcc_cubiccell", "wigner_bcc"]:
            smeareff = smearing * np.sqrt(3) / 2
        elif crystal_name == "wigner_sc":
            smeareff = smearing
        smeareff *= scaling_factor

        # Compute potential and compare against reference
        EP = EwaldPotential(atomic_smearing=smeareff)
        potentials = EP.compute(types, positions, cell, charges)
        energies = potentials * charges
        energies_ref = -torch.ones_like(energies) * madelung_ref
        torch.testing.assert_close(energies, energies_ref, atol=0.0, rtol=rtol)


orthogonal_transformations = generate_orthogonal_transformations()
scaling_factors = torch.tensor([0.4325, 2.0353610], dtype=dtype)


@pytest.mark.parametrize("sr_cutoff", [2.01, 5.5])
@pytest.mark.parametrize("frame_index", [0, 1, 2])
@pytest.mark.parametrize("scaling_factor", scaling_factors)
@pytest.mark.parametrize("ortho", orthogonal_transformations)
@pytest.mark.parametrize("calc_name", ["ewald", "pme"])
def test_random_structure(sr_cutoff, frame_index, scaling_factor, ortho, calc_name):
    """
    Check that the potentials obtained from the main code agree with the ones computed
    using an external library (GROMACS) for more complicated structures consisting of
    8 atoms placed randomly in cubic cells of varying sizes.
    """
    # Get the predefined frames with the
    # Coulomb energy and forces computed by GROMACS using PME
    # using parameters as defined in the GROMACS manual
    # https://manual.gromacs.org/documentation/current/user-guide/mdp-options.html#ewald
    #
    # coulombtype = PME
    # fourierspacing = 0.01  ; 1/nm
    # pme_order = 8
    # rcoulomb = 0.3  ; nm
    struc_path = "tests/reference_structures/"
    frame = read(os.path.join(struc_path, "coulomb_test_frames.xyz"), frame_index)

    # Energies in Gaussian units (without e²/[4 π ɛ_0] prefactor)
    energy_target = torch.tensor(frame.info["energy"], dtype=dtype) / scaling_factor
    # Forces in Gaussian units per Å
    forces_target = (
        torch.tensor(frame.arrays["forces"], dtype=dtype) / scaling_factor**2
    )

    # Convert into input format suitable for MeshLODE
    positions = scaling_factor * (torch.tensor(frame.positions, dtype=dtype) @ ortho)
    positions.requires_grad = True
    cell = scaling_factor * torch.tensor(np.array(frame.cell), dtype=dtype) @ ortho
    charges = torch.tensor([1, 1, 1, 1, -1, -1, -1, -1], dtype=dtype).reshape((-1, 1))
    types = torch.tensor([1, 1, 1, 1, 2, 2, 2, 2])

    # Compute potential using MeshLODE and compare against reference values
    sr_cutoff = scaling_factor * torch.tensor(sr_cutoff, dtype=dtype)
    if calc_name == "ewald":
        calc = EwaldPotential(sr_cutoff=sr_cutoff)
        rtol_e = 2e-5
        rtol_f = 3.6e-3
    elif calc_name == "pme":
        calc = MeshEwaldPotential(sr_cutoff=sr_cutoff)
        rtol_e = 4.5e-3  # 1.5e-3
        rtol_f = 2.5e-3  # 6e-3
    potentials = calc.compute(types, positions=positions, cell=cell, charges=charges)

    # Compute energy, taking into account the double counting of each pair
    energy = torch.sum(potentials * charges) / 2
    torch.testing.assert_close(energy, energy_target, atol=0.0, rtol=rtol_e)

    # Compute forces
    energy.backward()
    forces = -positions.grad
    torch.testing.assert_close(forces, forces_target @ ortho, atol=0.0, rtol=rtol_f)