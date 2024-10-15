import math
import os

import numpy as np
import pytest
import torch

# Imports for random structure
from ase.io import read
from utils import neighbor_list_torch

from torchpme import EwaldCalculator, InversePowerLawPotential, PMECalculator

DTYPE = torch.float64


def generate_orthogonal_transformations():
    # Generate rotation matrix along x-axis
    def rot_x(phi):
        rot = torch.zeros((3, 3), dtype=DTYPE)
        rot[0, 0] = rot[1, 1] = math.cos(phi)
        rot[0, 1] = -math.sin(phi)
        rot[1, 0] = math.sin(phi)
        rot[2, 2] = 1.0

        return rot

    # Generate rotation matrix along z-axis
    def rot_z(theta):
        rot = torch.zeros((3, 3), dtype=DTYPE)
        rot[0, 0] = rot[2, 2] = math.cos(theta)
        rot[0, 2] = math.sin(theta)
        rot[2, 0] = -math.sin(theta)
        rot[1, 1] = 1.0

        return rot

    # Generate a few rotation matrices
    rot_1 = rot_z(0.987654)
    rot_2 = rot_z(1.23456) @ rot_x(0.82321)
    transformations = [rot_1, rot_2]

    # make sure that the generated transformations are indeed orthogonal
    for q in transformations:
        id = torch.eye(3, dtype=DTYPE)
        id_2 = q.T @ q
        torch.testing.assert_close(id, id_2, atol=1e-15, rtol=1e-15)

    return transformations


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
        positions = torch.tensor([[0, 0, 0], [0.5, 0.5, 0.5]], dtype=DTYPE)
        charges = torch.tensor([-1.0, 1.0], dtype=DTYPE)
        cell = torch.eye(3, dtype=DTYPE)
        madelung_ref = 2.035361
        num_formula_units = 1

    # Sodium-Chloride (NaCl) structure using a primitive unit cell
    # - non-cubic unit cell (fcc)
    # - 1 atom pair in the unit cell
    # - Cation-Anion ratio of 1:1
    elif crystal_name == "NaCl_primitive":
        positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=DTYPE)
        charges = torch.tensor([1.0, -1.0], dtype=DTYPE)
        cell = torch.tensor([[0, 1.0, 1], [1, 0, 1], [1, 1, 0]], dtype=DTYPE)  # fcc
        madelung_ref = 1.74756
        num_formula_units = 1

    # Sodium-Chloride (NaCl) structure using a cubic unit cell
    # - cubic unit cell
    # - 4 atom pairs in the unit cell
    # - Cation-Anion ratio of 1:1
    elif crystal_name == "NaCl_cubic":
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
            dtype=DTYPE,
        )
        charges = torch.tensor([+1.0, -1, -1, -1, +1, +1, +1, -1], dtype=DTYPE)
        cell = 2 * torch.eye(3, dtype=DTYPE)
        madelung_ref = 1.747565
        num_formula_units = 4

    # ZnS (zincblende) structure
    # - non-cubic unit cell (fcc)
    # - 1 atom pair in the unit cell
    # - Cation-Anion ratio of 1:1
    # Remarks: we use a primitive unit cell which makes the lattice parameter of the
    # cubic cell equal to 2.
    elif crystal_name == "zincblende":
        positions = torch.tensor([[0, 0, 0], [0.5, 0.5, 0.5]], dtype=DTYPE)
        charges = torch.tensor([1.0, -1], dtype=DTYPE)
        cell = torch.tensor([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=DTYPE)
        madelung_ref = 2 * 1.63806 / np.sqrt(3)
        num_formula_units = 1

    # Wurtzite structure
    # - non-cubic unit cell (triclinic)
    # - 2 atom pairs in the unit cell
    # - Cation-Anion ratio of 1:1
    elif crystal_name == "wurtzite":
        u = 3 / 8
        c = np.sqrt(1 / u)
        positions = torch.tensor(
            [
                [0.5, 0.5 / np.sqrt(3), 0.0],
                [0.5, 0.5 / np.sqrt(3), u * c],
                [0.5, -0.5 / np.sqrt(3), 0.5 * c],
                [0.5, -0.5 / np.sqrt(3), (0.5 + u) * c],
            ],
            dtype=DTYPE,
        )
        charges = torch.tensor([1.0, -1, 1, -1], dtype=DTYPE)
        cell = torch.tensor(
            [[0.5, -0.5 * np.sqrt(3), 0], [0.5, 0.5 * np.sqrt(3), 0], [0, 0, c]],
            dtype=DTYPE,
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
        positions = a * torch.tensor(
            [[1 / 4, 1 / 4, 1 / 4], [3 / 4, 3 / 4, 3 / 4], [0, 0, 0]], dtype=DTYPE
        )
        charges = torch.tensor([-1, -1, 2], dtype=DTYPE)
        cell = torch.tensor([[a, a, 0], [a, 0, a], [0, a, a]], dtype=DTYPE) / 2.0
        madelung_ref = 11.636575
        num_formula_units = 1

    # Copper(I)-Oxide structure (e.g. Cu2O with Cu+ and O2-)
    # - cubic unit cell
    # - 2 neutral molecules per unit cell
    # - Cation-Anion ratio of 2:1
    elif crystal_name == "cu2o":
        a = 1.0
        positions = a * torch.tensor(
            [
                [0, 0, 0],
                [1 / 2, 1 / 2, 1 / 2],
                [1 / 4, 1 / 4, 1 / 4],
                [1 / 4, 3 / 4, 3 / 4],
                [3 / 4, 1 / 4, 3 / 4],
                [3 / 4, 3 / 4, 1 / 4],
            ],
            dtype=DTYPE,
        )
        charges = torch.tensor([-2, -2, 1, 1, 1, 1], dtype=DTYPE)
        cell = a * torch.eye(3, dtype=DTYPE)
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
        positions = torch.tensor([[0, 0, 0]], dtype=DTYPE)
        charges = torch.tensor([1.0], dtype=DTYPE)
        cell = torch.tensor([[1.0, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=DTYPE)

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
        positions = torch.tensor([[0, 0, 0]], dtype=DTYPE)
        charges = torch.tensor([1.0], dtype=DTYPE)
        cell = torch.tensor(
            [[1.0, 0, 0], [0, 1, 0], [1 / 2, 1 / 2, 1 / 2]], dtype=DTYPE
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
        positions = torch.tensor([[0, 0, 0], [1 / 2, 1 / 2, 1 / 2]], dtype=DTYPE)
        charges = torch.tensor([1.0, 1.0], dtype=DTYPE)
        cell = torch.tensor([[1.0, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=DTYPE)

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
        positions = torch.tensor([[0, 0, 0]], dtype=DTYPE)
        charges = torch.tensor([1.0], dtype=DTYPE)
        cell = torch.tensor([[1, 0, 1], [0, 1, 1], [1, 1, 0]], dtype=DTYPE) / 2

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
        positions = 0.5 * torch.tensor(
            [[0.0, 0, 0], [1, 0, 1], [1, 1, 0], [0, 1, 1]], dtype=DTYPE
        )
        charges = torch.tensor([1.0, 1, 1, 1], dtype=DTYPE)
        cell = torch.eye(3, dtype=DTYPE)

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

    madelung_ref = torch.tensor(madelung_ref, dtype=DTYPE)
    charges = charges.reshape((-1, 1))

    return positions, charges, cell, madelung_ref, num_formula_units


@pytest.mark.parametrize("calc_name", ["ewald", "pme"])
@pytest.mark.parametrize(
    "crystal_name",
    [
        "CsCl",
        "NaCl_primitive",
        "NaCl_cubic",
        "zincblende",
        "wurtzite",
        "cu2o",
        "fluorite",
    ],
)
@pytest.mark.parametrize("scaling_factor", [1 / 2.0353610, 1.0, 3.4951291])
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
    pos, charges, cell, madelung_ref, num_units = define_crystal(crystal_name)
    pos *= scaling_factor
    cell *= scaling_factor
    madelung_ref /= scaling_factor
    charges = charges.reshape((-1, 1))

    # Define calculator and tolerances
    if calc_name == "ewald":
        sr_cutoff = scaling_factor
        smearing = sr_cutoff / 5.0
        lr_wavelength = 0.5 * smearing
        calc = EwaldCalculator(
            InversePowerLawPotential(
                exponent=1.0,
                smearing=smearing,
            ),
            lr_wavelength=lr_wavelength,
        )
        rtol = 4e-6
    elif calc_name == "pme":
        sr_cutoff = 2 * scaling_factor
        smearing = sr_cutoff / 5.0
        calc = PMECalculator(
            InversePowerLawPotential(
                exponent=1.0,
                smearing=smearing,
            ),
            mesh_spacing=smearing / 8,
        )
        rtol = 9e-4

    # Compute neighbor list
    neighbor_indices, neighbor_distances = neighbor_list_torch(
        positions=pos, periodic=True, box=cell, cutoff=sr_cutoff
    )

    # Compute potential and compare against target value using default hypers
    calc.to(dtype=DTYPE)
    potentials = calc.forward(
        positions=pos,
        charges=charges,
        cell=cell,
        neighbor_indices=neighbor_indices,
        neighbor_distances=neighbor_distances,
    )
    energies = potentials * charges
    madelung = -torch.sum(energies) / num_units

    torch.testing.assert_close(madelung, madelung_ref, atol=0.0, rtol=rtol)


# Since structures without charge neutrality show slower convergence, these
# structures are tested separately.
@pytest.mark.parametrize(
    "crystal_name",
    [
        "wigner_sc",
        "wigner_fcc",
        "wigner_fcc_cubiccell",
        "wigner_bcc",
        "wigner_bcc_cubiccell",
    ],
)
@pytest.mark.parametrize("scaling_factor", [0.4325, 1.0, 2.0353610])
def test_wigner(crystal_name, scaling_factor):
    """
    Check that the energy of a Wigner solid obtained from the Ewald sum calculator
    matches the reference values.
    In this test, the Wigner solids are defined by placing arranging positively charged
    point particles on a bcc lattice, leading to a net charge of the unit cell if we
    only look at the ions. This charge is compensated by a homogeneous neutral back-
    ground charge of opposite sign (physically: completely delocalized electrons).

    The presence of a net charge (due to the particles but without background) leads
    to numerically slower convergence of the relevant sums.
    """
    # Get parameters defining atomic positions, cell and charges
    positions, charges, cell, madelung_ref, _ = define_crystal(crystal_name)
    positions *= scaling_factor
    cell *= scaling_factor
    madelung_ref /= scaling_factor

    # Compute neighbor list
    neighbor_indices, neighbor_distances = neighbor_list_torch(
        positions=positions, periodic=True, box=cell
    )

    # The first value of 0.1 corresponds to what would be
    # chosen by default for the "wigner_sc" or "wigner_bcc_cubiccell" structure.
    smearings = torch.tensor([0.1, 0.06, 0.019], dtype=torch.float64)
    for smearing in smearings:
        # Readjust smearing parameter to match nearest neighbor distance
        if crystal_name in ["wigner_fcc", "wigner_fcc_cubiccell"]:
            smeareff = float(smearing) / np.sqrt(2)
        elif crystal_name in ["wigner_bcc_cubiccell", "wigner_bcc"]:
            smeareff = float(smearing) * np.sqrt(3) / 2
        elif crystal_name == "wigner_sc":
            smeareff = float(smearing)
        smeareff *= scaling_factor

        # Compute potential and compare against reference
        calc = EwaldCalculator(
            InversePowerLawPotential(
                exponent=1.0,
                smearing=smeareff,
            ),
            lr_wavelength=smeareff / 2,
        )
        calc.to(dtype=DTYPE)
        potentials = calc.forward(
            positions=positions,
            charges=charges,
            cell=cell,
            neighbor_indices=neighbor_indices,
            neighbor_distances=neighbor_distances,
        )
        energies = potentials * charges
        energies_ref = -torch.ones_like(energies) * madelung_ref / 2
        torch.testing.assert_close(energies, energies_ref, atol=0.0, rtol=4.2e-6)


@pytest.mark.parametrize("sr_cutoff", [5.54, 6.01])
@pytest.mark.parametrize("frame_index", [0, 1, 2])
@pytest.mark.parametrize("scaling_factor", [0.4325, 1.3353610])
@pytest.mark.parametrize("ortho", generate_orthogonal_transformations())
@pytest.mark.parametrize("calc_name", ["ewald", "pme"])
@pytest.mark.parametrize("full_neighbor_list", [True, False])
def test_random_structure(
    sr_cutoff, frame_index, scaling_factor, ortho, calc_name, full_neighbor_list
):
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
    energy_target = (
        torch.tensor(frame.get_potential_energy(), dtype=DTYPE) / scaling_factor
    )
    # Forces in Gaussian units per Å
    forces_target = torch.tensor(frame.get_forces(), dtype=DTYPE) / scaling_factor**2

    # Convert into input format suitable for torch-pme
    positions = scaling_factor * (torch.tensor(frame.positions, dtype=DTYPE) @ ortho)

    # Enable backward for positions
    positions.requires_grad = True

    cell = scaling_factor * torch.tensor(np.array(frame.cell), dtype=DTYPE) @ ortho
    charges = torch.tensor([1, 1, 1, 1, -1, -1, -1, -1], dtype=DTYPE).reshape((-1, 1))
    sr_cutoff = scaling_factor * sr_cutoff
    smearing = sr_cutoff / 6.0

    # Compute neighbor list
    neighbor_indices, neighbor_distances = neighbor_list_torch(
        positions=positions,
        periodic=True,
        box=cell,
        cutoff=sr_cutoff,
        full_neighbor_list=full_neighbor_list,
    )

    # Compute potential using torch-pme and compare against reference values
    if calc_name == "ewald":
        lr_wavelength = 0.5 * smearing
        calc = EwaldCalculator(
            InversePowerLawPotential(
                exponent=1.0,
                smearing=smearing,
            ),
            lr_wavelength=lr_wavelength,
            full_neighbor_list=full_neighbor_list,
        )
        rtol_e = 2e-5
        rtol_f = 3.5e-3
    elif calc_name == "pme":
        calc = PMECalculator(
            InversePowerLawPotential(
                exponent=1.0,
                smearing=smearing,
            ),
            mesh_spacing=smearing / 8,
            full_neighbor_list=full_neighbor_list,
        )
        rtol_e = 4.5e-3
        rtol_f = 5.0e-3

    calc.to(dtype=DTYPE)
    potentials = calc.forward(
        positions=positions,
        charges=charges,
        cell=cell,
        neighbor_indices=neighbor_indices,
        neighbor_distances=neighbor_distances,
    )

    # Compute energy. The double counting of the pairs is already taken into account.
    energy = torch.sum(potentials * charges)
    torch.testing.assert_close(energy, energy_target, atol=0.0, rtol=rtol_e)

    # Compute forces
    energy.backward()
    forces = -positions.grad
    torch.testing.assert_close(forces, forces_target @ ortho, atol=0.0, rtol=rtol_f)
