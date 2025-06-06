"""Workflow tests for the metatensor interface."""

import io
import sys
from pathlib import Path

import pytest
import torch
from packaging import version

import torchpme

sys.path.append(str(Path(__file__).parents[1]))
from helpers import DEVICES, DTYPES

mts_torch = pytest.importorskip("metatensor.torch")
mts_atomistic = pytest.importorskip("metatomic.torch")

SMEARING = 0.1
LR_WAVELENGTH = SMEARING / 4
MESH_SPACING = SMEARING / 4


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    ("CalculatorClass", "params"),
    [
        (
            torchpme.metatensor.Calculator,
            {
                "potential": torchpme.CoulombPotential(smearing=None),
            },
        ),
        (
            torchpme.metatensor.EwaldCalculator,
            {
                "potential": torchpme.CoulombPotential(smearing=SMEARING),
                "lr_wavelength": LR_WAVELENGTH,
            },
        ),
        (
            torchpme.metatensor.PMECalculator,
            {
                "potential": torchpme.CoulombPotential(smearing=SMEARING),
                "mesh_spacing": MESH_SPACING,
            },
        ),
        (
            torchpme.metatensor.P3MCalculator,
            {
                "potential": torchpme.CoulombPotential(smearing=SMEARING),
                "mesh_spacing": MESH_SPACING,
            },
        ),
    ],
)
class TestWorkflow:
    def system(self, device, dtype):
        system = mts_atomistic.System(
            types=torch.tensor([1, 2, 2]),
            positions=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.2], [0.0, 0.0, 0.5]]),
            cell=4.2 * torch.eye(3),
            pbc=torch.tensor([True, True, True]),
        )

        charges = torch.tensor([1.0, -0.5, -0.5]).unsqueeze(1)
        block = mts_torch.TensorBlock(
            values=charges,
            samples=mts_torch.Labels.range("atom", charges.shape[0]),
            components=[],
            properties=mts_torch.Labels.range("charge", charges.shape[1]),
        )

        tensor = mts_torch.TensorMap(
            keys=mts_torch.Labels(
                "_", torch.zeros(1, 1, dtype=torch.int32, device=device)
            ),
            blocks=[block],
        )

        system.add_data(name="charges", tensor=tensor)

        sample_values = torch.tensor([[0, 1, 0, 0, 0]])
        samples = mts_torch.Labels(
            names=[
                "first_atom",
                "second_atom",
                "cell_shift_a",
                "cell_shift_b",
                "cell_shift_c",
            ],
            values=sample_values,
        )

        values = torch.tensor([[[0.0], [0.0], [0.2]]])
        neighbors = mts_torch.TensorBlock(
            values=values,
            samples=samples,
            components=[mts_torch.Labels.range("xyz", 3)],
            properties=mts_torch.Labels.range("distance", 1),
        )

        return system.to(device=device, dtype=dtype), neighbors.to(
            device=device, dtype=dtype
        )

    def check_operation(self, calculator, device, dtype):
        """Make sure computation runs and returns a metatensor.TensorMap."""
        system, neighbors = self.system(device=device, dtype=dtype)
        descriptor = calculator.forward(system, neighbors)

        assert isinstance(descriptor, torch.ScriptObject)
        if version.parse(torch.__version__) >= version.parse("2.1"):
            assert descriptor._type().name() == "TensorMap"

    def test_operation_as_python(self, CalculatorClass, params, device, dtype):
        """Run `check_operation` as a normal python script"""
        calculator = CalculatorClass(**params)
        calculator.to(device=device, dtype=dtype)
        self.check_operation(calculator=calculator, device=device, dtype=dtype)

    def test_operation_as_torch_script(self, CalculatorClass, params, device, dtype):
        """Run `check_operation` as a compiled torch script module."""
        calculator = CalculatorClass(**params)
        calculator.to(device=device, dtype=dtype)
        scripted = torch.jit.script(calculator)
        self.check_operation(calculator=scripted, device=device, dtype=dtype)

    def test_save_load(self, CalculatorClass, params, device, dtype):
        """Save and load a compiled torch script module."""
        calculator = CalculatorClass(**params)
        calculator.to(device=device, dtype=dtype)
        scripted = torch.jit.script(calculator)
        with io.BytesIO() as buffer:
            torch.jit.save(scripted, buffer)
            buffer.seek(0)
            torch.jit.load(buffer)
