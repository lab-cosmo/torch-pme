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
    ("tune_mt", "tune_torch"),
    [
        (torchpme.metatensor.tune_ewald, torchpme.tune_ewald),
        (torchpme.metatensor.tune_pme, torchpme.tune_pme),
    ],
)
def test_tune(system, tune_mt, tune_torch):
    smearing_mts, params_mts, cutoff_mts = tune_mt(system)

    smearing_torch, params_torch, cutoff_torch = tune_torch(
        positions=system.positions,
        sum_squared_charges=torch.sum(system.get_data("charges").values ** 2, dim=0),
        cell=system.cell,
    )

    assert smearing_mts == smearing_torch
    assert params_mts == params_torch
    assert cutoff_mts == cutoff_torch
