import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from ase.io import read

from torchpme import EwaldCalculator, InversePowerLawPotential, PMECalculator

sys.path.append(str(Path(__file__).parents[1]))
from helpers import define_crystal, neighbor_list_torch

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
