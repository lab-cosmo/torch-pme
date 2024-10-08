"""
Madelung tests
"""

import io

import pytest
import torch
from packaging import version

import torchpme

mts_torch = pytest.importorskip("metatensor.torch")
mts_atomistic = pytest.importorskip("metatensor.torch.atomistic")

AVAILABLE_DEVICES = [torch.device("cpu")] + torch.cuda.is_available() * [
    torch.device("cuda")
]
SMEARING = 0.1
LR_WAVELENGTH = SMEARING / 4
MESH_SPACING = SMEARING / 4
NUM_NODES_PER_AXIS = 3


@pytest.mark.parametrize(
    "CalculatorClass, params",
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
                "interpolation_nodes": NUM_NODES_PER_AXIS,
            },
        ),
    ],
)
class TestWorkflow:
    def system(self, device=None):
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

        return system.to(device=device), neighbors.to(device=device)

    def check_operation(self, calculator, device):
        """Make sure computation runs and returns a metatensor.TensorMap."""
        system, neighbors = self.system(device)
        descriptor = calculator.forward(system, neighbors)

        assert isinstance(descriptor, torch.ScriptObject)
        if version.parse(torch.__version__) >= version.parse("2.1"):
            assert descriptor._type().name() == "TensorMap"

    @pytest.mark.parametrize("device", AVAILABLE_DEVICES)
    def test_operation_as_python(self, CalculatorClass, params, device):
        """Run `check_operation` as a normal python script"""
        calculator = CalculatorClass(**params)
        self.check_operation(calculator=calculator, device=device)

    @pytest.mark.parametrize("device", AVAILABLE_DEVICES)
    def test_operation_as_torch_script(self, CalculatorClass, params, device):
        """Run `check_operation` as a compiled torch script module."""
        calculator = CalculatorClass(**params)
        scripted = torch.jit.script(calculator)
        self.check_operation(calculator=scripted, device=device)

    def test_save_load(self, CalculatorClass, params):
        calculator = CalculatorClass(**params)
        scripted = torch.jit.script(calculator)
        with io.BytesIO() as buffer:
            torch.jit.save(scripted, buffer)
            buffer.seek(0)
            torch.jit.load(buffer)
