import pytest
import torch

import torchpme

mts_torch = pytest.importorskip("metatensor.torch")
mts_atomistic = pytest.importorskip("metatensor.torch.atomistic")


@pytest.fixture
def system():
    system = mts_atomistic.System(
        types=torch.tensor([1, 2, 2]),
        positions=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.2], [0.0, 0.0, 0.5]]),
        cell=4.2 * torch.eye(3),
    )

    charges = torch.tensor([1.0, -0.5, -0.5]).unsqueeze(1)
    data = mts_torch.TensorBlock(
        values=charges,
        samples=mts_torch.Labels.range("atom", charges.shape[0]),
        components=[],
        properties=mts_torch.Labels.range("charge", charges.shape[1]),
    )

    system.add_data(name="charges", data=data)

    return system


@pytest.mark.parametrize(
    "tune_mt",
    "tune_torch",
    [
        (torchpme.metatensor.tune_ewald, torchpme.tune_ewald),
        (torchpme.metatensor.tune_pme, torchpme.tune_pme),
    ],
)
def test_tune(system):
    smearing_mts, params_mts, cutoff_mts = torchpme.metatensor.tune_ewald(system)

    smearing_torch, params_torch, cutoff_torch = torchpme.tune_ewald(
        positions=system.positions,
        charges=system.get_data("charges").values,
        cell=system.cell,
    )

    assert smearing_mts == smearing_torch
    assert params_mts == params_torch
    assert cutoff_mts == cutoff_torch
