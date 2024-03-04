"""
Madelung tests
"""

import pytest
import torch
from torch.testing import assert_close


meshlode_metatensor = pytest.importorskip("meshlode.metatensor")
mts_atomistic = pytest.importorskip("metatensor.torch.atomistic")


class TestMadelung:
    """
    Test features computed in MeshPotential correspond to the "electrostatic" potential
    of the structures. We thus compare the computed potential against the known exact
    values for some simple crystal structures.
    """

    scaling_factors = torch.tensor([0.5, 1.2, 3.3])
    crystal_list = ["NaCl", "CsCl", "ZnS", "ZnSO4"]
    crystal_list_powers_of_2 = ["NaCl", "CsCl", "ZnS"]

    @pytest.fixture
    def crystal_dictionary(self):
        """
        Define the parameters of the three binary crystal structures:
        NaCl, CsCl and ZnCl. The reference values of the Madelung
        constants is taken from the book "Solid State Physics"
        by Ashcroft and Mermin.

        Note: Symbols and charges keys have to be sorted according to their
        atomic number in ascending alternating order! For an example see
        ZnS04 in the wurtzite structure.
        """
        # Initialize dictionary for crystal paramaters
        d = {k: {} for k in self.crystal_list}
        SQRT3 = torch.sqrt(torch.tensor(3))

        # NaCl structure
        # Using a primitive unit cell, the distance between the
        # closest Na-Cl pair is exactly 1. The cubic unit cell
        # in these units would have a length of 2.
        d["NaCl"]["symbols"] = ["Na", "Cl"]
        d["NaCl"]["types"] = torch.tensor([11, 17])
        d["NaCl"]["charges"] = torch.tensor([[1.0, -1]]).T
        d["NaCl"]["positions"] = torch.tensor([[0, 0, 0], [1.0, 0, 0]])
        d["NaCl"]["cell"] = torch.tensor([[0, 1.0, 1], [1, 0, 1], [1, 1, 0]])
        d["NaCl"]["madelung"] = 1.7476

        # CsCl structure
        # This structure is simple since the primitive unit cell
        # is just the usual cubic cell with side length set to one.
        # The closest Cs-Cl distance is sqrt(3)/2. We thus divide
        # the Madelung constant by this value to match the reference.
        d["CsCl"]["symbols"] = ["Cs", "Cl"]
        d["CsCl"]["types"] = torch.tensor([55, 17])
        d["CsCl"]["charges"] = torch.tensor([[1.0, -1]]).T
        d["CsCl"]["positions"] = torch.tensor([[0, 0, 0], [0.5, 0.5, 0.5]])
        d["CsCl"]["cell"] = torch.eye(3)
        d["CsCl"]["madelung"] = 2 * 1.7626 / SQRT3

        # ZnS (zincblende) structure
        # As for NaCl, a primitive unit cell is used which makes
        # the lattice parameter of the cubic cell equal to 2.
        # In these units, the closest Zn-S distance is sqrt(3)/2.
        # We thus divide the Madelung constant by this value.
        # If, on the other han_pylode_without_centerd, we set the lattice constant of
        # the cubic cell equal to 1, the Zn-S distance is sqrt(3)/4.
        d["ZnS"]["symbols"] = ["S", "Zn"]
        d["ZnS"]["types"] = torch.tensor([16, 30])
        d["ZnS"]["charges"] = torch.tensor([[1.0, -1]]).T
        d["ZnS"]["positions"] = torch.tensor([[0, 0, 0], [0.5, 0.5, 0.5]])
        d["ZnS"]["cell"] = torch.tensor([[0, 1.0, 1], [1, 0, 1], [1, 1, 0]])
        d["ZnS"]["madelung"] = 2 * 1.6381 / SQRT3

        # ZnS (O4) in wurtzite structure (triclinic cell)
        u = torch.tensor([3 / 8])
        c = torch.sqrt(1 / u)
        d["ZnSO4"]["symbols"] = ["S", "Zn", "S", "Zn"]
        d["ZnSO4"]["types"] = torch.tensor([16, 30, 16, 30])
        d["ZnSO4"]["charges"] = torch.tensor([[1.0, -1, 1, -1]]).T
        d["ZnSO4"]["positions"] = torch.tensor(
            [
                [0.5, 0.5 / SQRT3, 0.0],
                [0.5, 0.5 / SQRT3, u * c],
                [0.5, -0.5 / SQRT3, 0.5 * c],
                [0.5, -0.5 / SQRT3, (0.5 + u) * c],
            ]
        )
        d["ZnSO4"]["cell"] = torch.tensor(
            [[0.5, -0.5 * SQRT3, 0], [0.5, 0.5 * SQRT3, 0], [0, 0, c]]
        )

        d["ZnSO4"]["madelung"] = 1.6413 / (u * c)

        return d

    @pytest.mark.parametrize("crystal_name", crystal_list_powers_of_2)
    @pytest.mark.parametrize("atomic_smearing", [0.1, 0.05])
    @pytest.mark.parametrize("interpolation_order", [1, 2])
    @pytest.mark.parametrize("scaling_factor", scaling_factors)
    def test_madelung_low_order(
        self,
        crystal_dictionary,
        crystal_name,
        atomic_smearing,
        scaling_factor,
        interpolation_order,
    ):
        """
        For low interpolation orders, if the atoms already lie exactly on a mesh point,
        there are no additional errors due to atomic_smearing the charges. Thus, we can
        reach a relatively high accuracy.
        """
        dic = crystal_dictionary[crystal_name]
        positions = dic["positions"] * scaling_factor
        cell = dic["cell"] * scaling_factor
        charges = dic["charges"]
        madelung = dic["madelung"] / scaling_factor
        mesh_spacing = atomic_smearing / 2 * scaling_factor
        smearing_eff = atomic_smearing * scaling_factor
        MP = meshlode_metatensor.MeshPotential(
            smearing_eff, mesh_spacing, interpolation_order, subtract_self=True
        )
        potentials_mesh = MP._compute_single_system(
            positions=positions, charges=charges, cell=cell
        )
        energies = potentials_mesh * charges
        energies_target = -torch.ones_like(energies) * madelung
        assert_close(energies, energies_target, rtol=1e-4, atol=1e-6)

    @pytest.mark.parametrize("crystal_name", crystal_list)
    @pytest.mark.parametrize("atomic_smearing", [0.2, 0.12])
    @pytest.mark.parametrize("interpolation_order", [3, 4, 5])
    @pytest.mark.parametrize("scaling_factor", scaling_factors)
    def test_madelung_high_order(
        self,
        crystal_dictionary,
        crystal_name,
        atomic_smearing,
        scaling_factor,
        interpolation_order,
    ):
        """
        For high interpolation order, the current naive implementation used to subtract
        the center contribution introduces additional errors since an atom is smeared
        onto multiple mesh points, turning the short-range correction into a more
        complicated expression that has not yet been implemented. Thus, we use a much
        larger tolerance of 1e-2 for the precision needed in the calculation.
        """
        dic = crystal_dictionary[crystal_name]
        positions = dic["positions"] * scaling_factor
        cell = dic["cell"] * scaling_factor
        charges = dic["charges"]
        madelung = dic["madelung"] / scaling_factor
        mesh_spacing = atomic_smearing / 10 * scaling_factor
        smearing_eff = atomic_smearing * scaling_factor
        MP = meshlode_metatensor.MeshPotential(
            smearing_eff, mesh_spacing, interpolation_order, subtract_self=True
        )
        potentials_mesh = MP._compute_single_system(
            positions=positions, charges=charges, cell=cell
        )
        energies = potentials_mesh * charges
        energies_target = -torch.ones_like(energies) * madelung
        assert_close(energies, energies_target, rtol=1e-2, atol=1e-3)

    @pytest.mark.parametrize("crystal_name", crystal_list_powers_of_2)
    @pytest.mark.parametrize("atomic_smearing", [0.1, 0.05])
    @pytest.mark.parametrize("interpolation_order", [1, 2])
    @pytest.mark.parametrize("scaling_factor", scaling_factors)
    def test_madelung_low_order_metatensor(
        self,
        crystal_dictionary,
        crystal_name,
        atomic_smearing,
        scaling_factor,
        interpolation_order,
    ):
        """
        Same test as above but now using the main compute function of the class that is
        actually facing the user and outputting in metatensor format.
        """
        dic = crystal_dictionary[crystal_name]
        positions = dic["positions"] * scaling_factor
        cell = dic["cell"] * scaling_factor
        types = dic["types"]
        charges = dic["charges"]
        madelung = dic["madelung"] / scaling_factor
        mesh_spacing = atomic_smearing / 2 * scaling_factor
        smearing_eff = atomic_smearing * scaling_factor
        n_atoms = len(positions)
        system = mts_atomistic.System(types=types, positions=positions, cell=cell)
        MP = meshlode_metatensor.MeshPotential(
            atomic_smearing=smearing_eff,
            mesh_spacing=mesh_spacing,
            interpolation_order=interpolation_order,
            subtract_self=True,
        )
        potentials_mesh = MP.compute(system)

        # Compute the actual potential from the features
        energies = torch.zeros((n_atoms, 1))
        for idx_c, c in enumerate(types):
            for idx_n, n in enumerate(types):
                block = potentials_mesh.block(
                    {"center_type": int(c), "neighbor_type": int(n)}
                )
                energies[idx_c] += charges[idx_c] * charges[idx_n] * block.values[0, 0]

        energies_ref = -madelung * torch.ones((n_atoms, 1))
        assert_close(energies, energies_ref, rtol=1e-4, atol=1e-6)
