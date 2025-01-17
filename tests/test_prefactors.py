from scipy import constants

from torchpme import prefactors


def test_units():
    eps = 1e-8
    gauss_to_si = constants.elementary_charge**2 / (
        4 * constants.pi * constants.epsilon_0
    )
    gauss_to_ev_per_A = gauss_to_si / (constants.elementary_charge * constants.angstrom)
    gauss_to_kcalmol_per_A = gauss_to_si / (
        constants.kilo * constants.calorie / constants.Avogadro * constants.angstrom
    )
    gauss_to_kjmol_per_A = gauss_to_si / (
        constants.kilo / constants.Avogadro * constants.angstrom
    )

    assert abs(prefactors.SI - gauss_to_si) / gauss_to_si < eps
    assert abs(prefactors.eV_A - gauss_to_ev_per_A) / gauss_to_ev_per_A < eps
    assert (
        abs(prefactors.kcalmol_A - gauss_to_kcalmol_per_A) / gauss_to_kcalmol_per_A
        < eps
    )
    assert abs(prefactors.kJmol - gauss_to_kjmol_per_A) / gauss_to_kjmol_per_A < eps
