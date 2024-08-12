"""
Madelung tests
"""

import io

import pytest
import torch
from packaging import version
from utils_metatensor import add_neighbor_list

import torchpme


mts_torch = pytest.importorskip("metatensor.torch")
mts_atomistic = pytest.importorskip("metatensor.torch.atomistic")

AVAILABLE_DEVICES = [torch.device("cpu")] + torch.cuda.is_available() * [
    torch.device("cuda")
]
ATOMIC_SMEARING = 0.1
LR_WAVELENGTH = ATOMIC_SMEARING / 4
MESH_SPACING = ATOMIC_SMEARING / 4
INTERPOLATION_ORDER = 2
SUBTRACT_SELF = True


@pytest.mark.parametrize(
    "CalculatorClass, params",
    [
        (torchpme.metatensor.DirectPotential, {}),
        (
            torchpme.metatensor.EwaldPotential,
            {
                "atomic_smearing": ATOMIC_SMEARING,
                "lr_wavelength": LR_WAVELENGTH,
                "subtract_self": SUBTRACT_SELF,
            },
        ),
        (
            torchpme.metatensor.PMEPotential,
            {
                "atomic_smearing": ATOMIC_SMEARING,
                "mesh_spacing": MESH_SPACING,
                "interpolation_order": INTERPOLATION_ORDER,
                "subtract_self": SUBTRACT_SELF,
            },
        ),
    ],
)
class TestWorkflow:
    def cscl_system(self, device=None):
        """CsCl crystal. Same as in the madelung test"""

        if device is None:
            device = torch.device("cpu")

        system = mts_atomistic.System(
            types=torch.tensor([17, 55]),
            positions=torch.tensor([[0, 0, 0], [0.5, 0.5, 0.5]]),
            cell=torch.eye(3),
        )

        data = mts_torch.TensorBlock(
            values=torch.tensor([-1.0, 1.0]).reshape(-1, 1),
            samples=mts_torch.Labels.range("atom", len(system)),
            components=[],
            properties=mts_torch.Labels("charge", torch.tensor([[0]])),
        )
        system.add_data(name="charges", data=data)
        add_neighbor_list(system)

        return system.to(device=device)

    def check_operation(self, calculator, device):
        """Make sure computation runs and returns a metatensor.TensorMap."""
        descriptor_compute = calculator.compute(self.cscl_system(device))
        descriptor_forward = calculator.forward(self.cscl_system(device))

        assert isinstance(descriptor_compute, torch.ScriptObject)
        assert isinstance(descriptor_forward, torch.ScriptObject)
        if version.parse(torch.__version__) >= version.parse("2.1"):
            assert descriptor_compute._type().name() == "TensorMap"
            assert descriptor_forward._type().name() == "TensorMap"

        assert mts_torch.equal(descriptor_forward, descriptor_compute)

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
