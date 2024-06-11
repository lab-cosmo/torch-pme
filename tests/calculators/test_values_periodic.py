import numpy as np
import pytest
import torch

from meshlode import EwaldPotential


def define_crystal(crystal_name="CsCl"):
    # Define all relevant parameters (atom positions, charges, cell) of the reference
    # crystal structures for which the Madelung constants obtained from the Ewald sums
    # are compared with reference values.
    # see https://www.sciencedirect.com/science/article/pii/B9780128143698000078#s0015
    # More detailed values can be found in https://pubs.acs.org/doi/10.1021/ic2023852
    dtype = torch.float64

    # Caesium-Chloride (CsCl) structure:
    # - Cubic unit cell
    # - 1 atom pair in the unit cell
    # - Cation-Anion ratio of 1:1
    if crystal_name == "CsCl":
        types = torch.tensor([17, 55])  # Cl and Cs
        positions = torch.tensor([[0, 0, 0], [0.5, 0.5, 0.5]], dtype=dtype)
        charges = torch.tensor([-1.0, 1.0], dtype=dtype)
        cell = torch.eye(3, dtype=dtype)
        madelung_reference = 2.035361

    # Sodium-Chloride (NaCl) structure using a primitive unit cell
    # - non-cubic unit cell (fcc)
    # - 1 atom pair in the unit cell
    # - Cation-Anion ratio of 1:1
    elif crystal_name == "NaCl_primitive":
        types = torch.tensor([11, 17])
        positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=dtype)
        charges = torch.tensor([1.0, -1.0], dtype=dtype)
        cell = torch.tensor([[0, 1.0, 1], [1, 0, 1], [1, 1, 0]], dtype=dtype)  # fcc
        madelung_reference = 1.74756

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
        madelung_reference = 1.747565

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
        madelung_reference = 2 * 1.63806 / np.sqrt(3)

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
        madelung_reference = 1.64132 / (u * c)

    # Fluorite structure
    # - non-cubic (fcc) unit cell
    # - 1 neutral molecule per unit cell
    # - Cation-Anion ratio of 2:1
    elif crystal_name == "fluorite":
        a = 5.463
        a = 1.0
        types = torch.tensor([9, 9, 20])
        positions = a * torch.tensor(
            [[1 / 4, 1 / 4, 1 / 4], [3 / 4, 3 / 4, 3 / 4], [0, 0, 0]], dtype=dtype
        )
        charges = torch.tensor([-1, -1, 2], dtype=dtype)
        cell = torch.tensor([[a, a, 0], [a, 0, a], [0, a, a]], dtype=dtype) / 2.0
        madelung_reference = 11.636575

    # Copper-Oxide Cu2O structure
    elif crystal_name == "cu2o":
        a = 0.4627
        a = 1.0
        types = torch.tensor([8, 29, 29])
        positions = a * torch.tensor(
            [[1 / 4, 1 / 4, 1 / 4], [0, 0, 0], [1 / 2, 1 / 2, 1 / 2]], dtype=dtype
        )
        charges = torch.tensor([-2, 1, 1], dtype=dtype)
        cell = torch.tensor([[a, 0, 0], [0, a, 0], [0, 0, a]], dtype=dtype)
        madelung_reference = 10.2594570330750

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
        madelung_reference = madelung_wigner_seiz / wigner_seiz_radius  # 2.83730

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
        madelung_reference = madelung_wigner_seiz / wigner_seiz_radius  # 3.63924

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
        madelung_reference = madelung_wigner_seiz / wigner_seiz_radius  # 3.63924

    # Wigner crystal in fcc structure
    # See description of "wigner_sc" for a general explanation on Wigner crystals.
    # Used to test the code for cases in which the unit cell has a nonzero net charge.
    elif crystal_name == "wigner_fcc":
        types = torch.tensor([1])
        positions = torch.tensor([[0.0, 0, 0]], dtype=dtype)
        charges = torch.tensor([1.0], dtype=dtype)
        cell = torch.tensor([[1, 0, 1], [0, 1, 1], [1, 1, 0]], dtype=dtype) / 2

        # Reference value is expressed in terms of the Wigner-Seiz radius, and needs to
        # be rescaled to the case in which the lattice parameter = 1.
        madelung_wigner_seiz = 1.791753
        wigner_seiz_radius = (3 / (4 * np.pi * 4)) ** (
            1 / 3
        )  # 4 atoms per cubic unit cell
        madelung_reference = madelung_wigner_seiz / wigner_seiz_radius  # 4.58488

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
        madelung_reference = madelung_wigner_seiz / wigner_seiz_radius  # 4.58488

    else:
        raise ValueError(f"crystal_name = {crystal_name} is not supported!")

    return types, positions, charges, cell, madelung_reference


neutral_crystals = ["CsCl", "NaCl_primitive", "NaCl_cubic", "zincblende", "wurtzite"]
# neutral_crystals = ['CsCl']
scaling_factors = torch.tensor([1 / 2.0353610, 1.0, 3.4951291], dtype=torch.float64)
@pytest.mark.parametrize("crystal_name", neutral_crystals)
@pytest.mark.parametrize("scaling_factor", scaling_factors)
def test_madelung(crystal_name, scaling_factor):
    """
    Check that the Madelung constants obtained from the Ewald sum calculator matches
    the reference values.
    In this test, only the charge-neutral crystal systems are chosen for which the
    potential converges relatively quickly, while the systems with a net charge are
    treated separately below.
    """
    # Call Ewald potential class without specifying any of the convergence parameters
    # so that they are chosen by default (in a structure-dependent way)
    EP = EwaldPotential()

    # Compute potential at the position of the atoms for the specified structure
    types, positions, charges, cell, madelung_reference = define_crystal(crystal_name)
    positions *= scaling_factor
    cell *= scaling_factor
    potentials = EP.compute(types, positions, cell, charges)
    energies = potentials * charges
    energies_ref = -torch.ones_like(energies) * madelung_reference / scaling_factor

    torch.testing.assert_close(energies, energies_ref, atol=0.0, rtol=3.1e-6)


wigner_crystals = [
    "wigner_sc",
    "wigner_fcc",
    "wigner_fcc_cubiccell",
    "wigner_bcc",
    "wigner_bcc_cubiccell",
]
wigner_crystal = ['wigner_sc']
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
    types, positions, charges, cell, madelung_reference = define_crystal(crystal_name)
    positions *= scaling_factor
    cell *= scaling_factor
    madelung_reference /= scaling_factor

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
        energies_ref = -torch.ones_like(energies) * madelung_reference
        torch.testing.assert_close(energies, energies_ref, atol=0.0, rtol=rtol)
