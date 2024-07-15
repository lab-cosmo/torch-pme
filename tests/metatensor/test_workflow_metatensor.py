"""
Madelung tests
"""

import io

import pytest
import torch
from packaging import version
from utils_metatensor import add_neighbor_list

import meshlode


mts_torch = pytest.importorskip("metatensor.torch")
mts_atomistic = pytest.importorskip("metatensor.torch.atomistic")


ATOMIC_SMEARING = 0.1
LR_WAVELENGTH = ATOMIC_SMEARING / 4
MESH_SPACING = ATOMIC_SMEARING / 4
INTERPOLATION_ORDER = 2
SUBTRACT_SELF = True


@pytest.mark.parametrize(
    "CalculatorClass, params",
    [
        (meshlode.metatensor.DirectPotential, {}),
        (
            meshlode.metatensor.EwaldPotential,
            {
                "atomic_smearing": ATOMIC_SMEARING,
                "lr_wavelength": LR_WAVELENGTH,
                "subtract_self": SUBTRACT_SELF,
            },
        ),
        (
            meshlode.metatensor.PMEPotential,
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
    def cscl_system(self):
        """CsCl crystal. Same as in the madelung test"""

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

        return system

    def check_operation(self, calculator):
        """Make sure computation runs and returns a metatensor.TensorMap."""
        descriptor_compute = calculator.compute(self.cscl_system())
        descriptor_forward = calculator.forward(self.cscl_system())

        assert isinstance(descriptor_compute, torch.ScriptObject)
        assert isinstance(descriptor_forward, torch.ScriptObject)
        if version.parse(torch.__version__) >= version.parse("2.1"):
            assert descriptor_compute._type().name() == "TensorMap"
            assert descriptor_forward._type().name() == "TensorMap"

        assert mts_torch.equal(descriptor_forward, descriptor_compute)

    def test_operation_as_python(self, CalculatorClass, params):
        """Run `check_operation` as a normal python script"""
        calculator = CalculatorClass(**params)
        self.check_operation(calculator)

    def test_operation_as_torch_script(self, CalculatorClass, params):
        """Run `check_operation` as a compiled torch script module."""
        calculator = CalculatorClass(**params)
        scripted = torch.jit.script(calculator)
        self.check_operation(scripted)

    def test_save_load(self, CalculatorClass, params):
        calculator = CalculatorClass(**params)
        scripted = torch.jit.script(calculator)
        with io.BytesIO() as buffer:
            torch.jit.save(scripted, buffer)
            buffer.seek(0)
            torch.jit.load(buffer)
