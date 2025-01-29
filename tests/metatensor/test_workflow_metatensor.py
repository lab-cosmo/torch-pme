"""Madelung tests"""

import io

import pytest
import torch
from packaging import version

import torchpme

mts_torch = pytest.importorskip("metatensor.torch")
mts_atomistic = pytest.importorskip("metatensor.torch.atomistic")

DEVICES = ["cpu", torch.device("cpu")] + torch.cuda.is_available() * ["cuda"]
DTYPES = [torch.float32, torch.float64]
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
                "potential": lambda dtype, device: torchpme.CoulombPotential(
                    smearing=None, dtype=dtype, device=device
                ),
            },
        ),
        (
            torchpme.metatensor.EwaldCalculator,
            {
                "potential": lambda dtype, device: torchpme.CoulombPotential(
                    smearing=SMEARING, dtype=dtype, device=device
                ),
                "lr_wavelength": LR_WAVELENGTH,
            },
        ),
        (
            torchpme.metatensor.PMECalculator,
            {
                "potential": lambda dtype, device: torchpme.CoulombPotential(
                    smearing=SMEARING, dtype=dtype, device=device
                ),
                "mesh_spacing": MESH_SPACING,
            },
        ),
        (
            torchpme.metatensor.P3MCalculator,
            {
                "potential": lambda dtype, device: torchpme.CoulombPotential(
                    smearing=SMEARING, dtype=dtype, device=device
                ),
                "mesh_spacing": MESH_SPACING,
            },
        ),
    ],
)
class TestWorkflow:
    def system(self, device=None, dtype=None):
        device = torch.get_default_device() if device is None else torch.device(device)
        dtype = torch.get_default_dtype() if dtype is None else dtype

        system = mts_atomistic.System(
            types=torch.tensor([1, 2, 2]),
            positions=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.2], [0.0, 0.0, 0.5]]),
            cell=4.2 * torch.eye(3),
            pbc=torch.tensor([True, True, True]),
        )

        charges = torch.tensor([1.0, -0.5, -0.5]).unsqueeze(1)
        data = mts_torch.TensorBlock(
            values=charges,
            samples=mts_torch.Labels.range("atom", charges.shape[0]),
            components=[],
            properties=mts_torch.Labels.range("charge", charges.shape[1]),
        )

        system.add_data(name="charges", data=data)

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
        params = params.copy()
        params["potential"] = params["potential"](dtype, device)
        calculator = CalculatorClass(**params, device=device, dtype=dtype)
        self.check_operation(calculator=calculator, device=device, dtype=dtype)

    def test_operation_as_torch_script(self, CalculatorClass, params, device, dtype):
        """Run `check_operation` as a compiled torch script module."""
        params = params.copy()
        params["potential"] = params["potential"](dtype, device)
        calculator = CalculatorClass(**params, device=device, dtype=dtype)
        scripted = torch.jit.script(calculator)
        self.check_operation(calculator=scripted, device=device, dtype=dtype)

    def test_save_load(self, CalculatorClass, params, device, dtype):
        params = params.copy()
        params["potential"] = params["potential"](dtype, device)

        calculator = CalculatorClass(**params, device=device, dtype=dtype)
        scripted = torch.jit.script(calculator)
        with io.BytesIO() as buffer:
            torch.jit.save(scripted, buffer)
            buffer.seek(0)
            torch.jit.load(buffer)
