import sys
from pathlib import Path

import pytest
import torch
from ase.io import read

from torchpme import CalculatorDipole, PotentialDipole
from torchpme.prefactors import eV_A

sys.path.append(str(Path(__file__).parents[1]))
from helpers import (
    DEVICES,
    DIPOLES_TEST_FRAMES,
    DTYPES,
    compute_distances,
    neighbor_list,
)

frames = read(DIPOLES_TEST_FRAMES, ":3")
cutoffs = [3.9986718930, 4.0000000000, 4.7363281250]
alphas = [0.8819831493, 0.8956299559, 0.7215211182]
energies = [frame.get_potential_energy() for frame in frames]
forces = [frame.get_forces() for frame in frames]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
class TestDipoles:
    def parallel_dipoles(self, device, dtype):
        """Parallel dipoles along the y-axis"""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 4.0, 0.0]],
            dtype=dtype,
            device=device,
        )
        dipoles = torch.tensor(
            [[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
            dtype=dtype,
            device=device,
        )
        cell = torch.tensor(
            [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
            dtype=dtype,
            device=device,
        )
        neighbor_indices = torch.tensor(
            [[0, 1], [1, 2], [0, 2]], dtype=torch.int64, device=device
        )
        neighbor_vectors = torch.tensor(
            [[0.0, 2.0, 0.0], [0.0, 2.0, 0.0], [0.0, 4.0, 0.0]],
            dtype=dtype,
            device=device,
        )

        return dipoles, cell, positions, neighbor_indices, neighbor_vectors

    def test_magnetostatics_direct(self, device, dtype):
        calculator = CalculatorDipole(
            potential=PotentialDipole(),
            full_neighbor_list=False,
        )
        calculator.to(device=device, dtype=dtype)
        pot = calculator(*self.parallel_dipoles(device=device, dtype=dtype))
        dipoles = self.parallel_dipoles(device=device, dtype=dtype)[0]
        result = (pot * dipoles).sum()
        print(result)
        expected_result = torch.tensor(
            -0.265625, dtype=dtype, device=device
        )  # analytical result
        torch.testing.assert_close(result, expected_result)

    @pytest.mark.parametrize(
        ("smearing", "sr_potential"),
        [
            (
                1e10,
                torch.tensor(-0.265625),
            ),  # analytical result, should coinside with the direct calculation when smearing -> inf
            (1e-10, torch.tensor(0.0000)),  # should be zero when smearing -> 0
        ],
    )
    def test_magnetostatics_sr(self, device, dtype, smearing, sr_potential):
        calculator = CalculatorDipole(
            potential=PotentialDipole(smearing=smearing),
            full_neighbor_list=False,
            lr_wavelength=1.0,
        )
        calculator.to(device=device, dtype=dtype)
        system = self.parallel_dipoles(device=device, dtype=dtype)
        pot = calculator._compute_rspace(
            dipoles=system[0], neighbor_indices=system[3], neighbor_vectors=system[4]
        )
        dipoles = self.parallel_dipoles(device=device, dtype=dtype)[0]
        result = (pot * dipoles).sum()
        expected_result = sr_potential.to(dtype=dtype, device=device)
        torch.testing.assert_close(result, expected_result)

    def test_magnetostatic_ewald(self, device, dtype):
        alpha = 1.0
        smearing = (
            1 / (2 * alpha**2)
        ) ** 0.5  # convert espressomd alpha to torch-pme smearing
        calculator = CalculatorDipole(
            potential=PotentialDipole(smearing=smearing),
            full_neighbor_list=False,
            lr_wavelength=0.1,  # this value is heuristic
        )
        calculator.to(device=device, dtype=dtype)
        pot = calculator(*self.parallel_dipoles(device=device, dtype=dtype))
        dipoles = self.parallel_dipoles(device=device, dtype=dtype)[0]
        result = (pot * dipoles).sum()
        # result is calculated using espressomd DipolarP3M with the same parameters and mesh
        # size 64
        expected_result = torch.tensor(-0.30848574939287954, dtype=dtype, device=device)
        torch.testing.assert_close(result, expected_result, atol=1e-6, rtol=1e-4)

    @pytest.mark.parametrize(
        ("frame", "cutoff", "alpha", "energy", "force"),
        zip(frames, cutoffs, alphas, energies, forces),
    )
    def test_magnetostatic_ewald_crystal(
        self, frame, cutoff, alpha, energy, force, device, dtype
    ):
        smearing = (
            1 / (2 * alpha**2)
        ) ** 0.5  # convert espressomd alpha to torch-pme smearing
        calc = CalculatorDipole(
            potential=PotentialDipole(smearing=smearing, prefactor=eV_A),
            full_neighbor_list=False,
            lr_wavelength=0.1,
        )
        calc.to(device=device, dtype=dtype)
        positions = torch.tensor(frame.get_positions(), dtype=dtype, device=device)
        dipoles = torch.tensor(frame.get_array("dipoles"), dtype=dtype, device=device)
        cell = torch.tensor(frame.get_cell().array, dtype=dtype, device=device)
        neighbor_indices, neighbor_shifts = neighbor_list(
            positions=positions,
            periodic=True,
            box=cell,
            cutoff=cutoff,
            full_neighbor_list=False,
            neighbor_shifts=True,
        )
        positions.requires_grad = True
        neighbor_distance_vectors = compute_distances(
            positions=positions,
            neighbor_indices=neighbor_indices,
            cell=cell,
            neighbor_shifts=neighbor_shifts,
            norm=False,
        )
        pot = calc(
            dipoles=dipoles,
            cell=cell,
            positions=positions,
            neighbor_indices=neighbor_indices,
            neighbor_vectors=neighbor_distance_vectors,
        )

        result = (pot * dipoles).sum()
        expected_result = torch.tensor(energy, dtype=dtype, device=device)
        torch.testing.assert_close(result, expected_result, atol=1e-5, rtol=1e-4)

        forces = -torch.autograd.grad(result, positions)[0]
        expected_forces = torch.tensor(force, dtype=dtype, device=device)
        torch.testing.assert_close(forces, expected_forces, atol=1e-5, rtol=1e-4)
