"""
Madelung tests
"""

import pytest
import torch
from packaging import version


mts_torch = pytest.importorskip("metatensor.torch")
mts_atomistic = pytest.importorskip("metatensor.torch.atomistic")
meshlode_metatensor = pytest.importorskip("meshlode.metatensor")


ATOMIC_SMEARING = 0.1
LR_WAVELENGTH = ATOMIC_SMEARING / 4
MESH_SPACING = ATOMIC_SMEARING / 4
INTERPOLATION_ORDER = 2
SUBTRACT_SELF = True


@pytest.mark.parametrize(
    "CalculatorClass, params",
    [
        (meshlode_metatensor.DirectPotential, {}),
        (
            meshlode_metatensor.EwaldPotential,
            {
                "atomic_smearing": ATOMIC_SMEARING,
                "lr_wavelength": LR_WAVELENGTH,
                "subtract_self": SUBTRACT_SELF,
            },
        ),
        (
            meshlode_metatensor.PMEPotential,
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

        return system

    def calculator(self, CalculatorClass, params):
        return CalculatorClass(**params)

    def test_forward(self, CalculatorClass, params):
        calculator = self.calculator(CalculatorClass, params)
        descriptor_compute = calculator.compute(self.cscl_system())
        descriptor_forward = calculator.forward(self.cscl_system())

        assert isinstance(descriptor_compute, torch.ScriptObject)
        assert isinstance(descriptor_forward, torch.ScriptObject)
        if version.parse(torch.__version__) >= version.parse("2.1"):
            assert descriptor_compute._type().name() == "TensorMap"
            assert descriptor_forward._type().name() == "TensorMap"

        assert mts_torch.equal(descriptor_forward, descriptor_compute)

    # Make sure that the calculators are computing the features without raising errors,
    # and returns the correct output format (TensorMap)
    def check_operation(self, CalculatorClass, params):
        calculator = self.calculator(CalculatorClass, params)
        descriptor = calculator.compute(self.cscl_system())

        assert isinstance(descriptor, torch.ScriptObject)
        if version.parse(torch.__version__) >= version.parse("2.1"):
            assert descriptor._type().name() == "TensorMap"

    # Run the above test as a normal python script
    def test_operation_as_python(self, CalculatorClass, params):
        self.check_operation(CalculatorClass, params)

    # Similar to the above, but also testing that the code can be compiled as a torch
    # script
    # def test_operation_as_torch_script(self, CalculatorClass, params):
    #     scripted = torch.jit.script(CalculatorClass, params)
    #     self.check_operation(scripted)
