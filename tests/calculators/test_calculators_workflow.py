"""Basic tests if the calculator works and is torch scriptable. Actual tests are done
for the metatensor calculator."""

import math

import pytest
import torch
from torch.testing import assert_close

from meshlode import DirectPotential, EwaldPotential, MeshPotential, PMEPotential


MADELUNG_CSCL = torch.tensor(2 * 1.7626 / math.sqrt(3))
CHARGES_CSCL = torch.tensor([1.0, -1.0])


ATOMIC_SMEARING = 0.1
LR_WAVELENGTH = ATOMIC_SMEARING / 4
MESH_SPACING = ATOMIC_SMEARING / 4
INTERPOLATION_ORDER = 2
SUBTRACT_SELF = True


@pytest.mark.parametrize(
    "CalculatorClass, params, periodic",
    [
        (DirectPotential, {}, False),
        (
            EwaldPotential,
            {
                "atomic_smearing": ATOMIC_SMEARING,
                "lr_wavelength": LR_WAVELENGTH,
                "subtract_self": SUBTRACT_SELF,
            },
            True,
        ),
        (
            PMEPotential,
            {
                "atomic_smearing": ATOMIC_SMEARING,
                "mesh_spacing": MESH_SPACING,
                "interpolation_order": INTERPOLATION_ORDER,
                "subtract_self": SUBTRACT_SELF,
            },
            True,
        ),
        (
            MeshPotential,
            {
                "atomic_smearing": ATOMIC_SMEARING,
                "mesh_spacing": MESH_SPACING,
                "interpolation_order": INTERPOLATION_ORDER,
                "subtract_self": SUBTRACT_SELF,
            },
            True,
        ),
    ],
)
class TestWorkflow:

    def cscl_system(self, periodic):
        """CsCl crystal. Same as in the madelung test"""
        types = torch.tensor([55, 17])
        positions = torch.tensor([[0, 0, 0], [0.5, 0.5, 0.5]])
        cell = torch.eye(3)

        if periodic:
            return types, positions, cell
        else:
            return types, positions

    def cscl_system_with_charges(self, periodic):
        """CsCl crystal with charges."""
        charges = torch.tensor([[0.0, 1.0], [1.0, 0]])
        return self.cscl_system(periodic) + (charges,)

    def calculator(self, CalculatorClass, periodic, params):
        if periodic:
            return CalculatorClass(**params)
        else:
            return CalculatorClass()

    def test_forward(self, CalculatorClass, periodic, params):
        calculator = self.calculator(CalculatorClass, periodic, params)
        descriptor_compute = calculator.compute(*self.cscl_system(periodic))
        descriptor_forward = calculator.forward(*self.cscl_system(periodic))

        assert type(descriptor_compute) is torch.Tensor
        assert type(descriptor_forward) is torch.Tensor
        assert torch.equal(descriptor_forward, descriptor_compute)

    def test_atomic_smearing_error(self, CalculatorClass, params, periodic):
        if periodic:
            with pytest.raises(ValueError, match="has to be positive"):
                CalculatorClass(atomic_smearing=-1.0)

    def test_interpolation_order_error(self, CalculatorClass, params, periodic):
        if type(CalculatorClass) in [PMEPotential, MeshPotential]:
            match = "Only `interpolation_order` from 1 to 5"
            with pytest.raises(ValueError, match=match):
                CalculatorClass(atomic_smearing=1, interpolation_order=10)

    def test_all_types(self, CalculatorClass, params, periodic):
        if periodic:
            descriptor = CalculatorClass(atomic_smearing=0.1, all_types=[8, 55, 17])
            values = descriptor.compute(*self.cscl_system(periodic))
            assert values.shape == (2, 3)
            assert torch.equal(values[:, 0], torch.zeros(2))

    def test_all_types_error(self, CalculatorClass, params, periodic):
        if periodic:
            descriptor = CalculatorClass(atomic_smearing=0.1, all_types=[17])
            with pytest.raises(ValueError, match="Global list of types"):
                descriptor.compute(*self.cscl_system(periodic))

    def test_single_frame(self, CalculatorClass, periodic, params):
        calculator = self.calculator(CalculatorClass, periodic, params)
        values = calculator.compute(*self.cscl_system(periodic))
        assert_close(
            MADELUNG_CSCL,
            CHARGES_CSCL[0] * values[0, 0] + CHARGES_CSCL[1] * values[0, 1],
            atol=1e4,
            rtol=1e-5,
        )

    def test_single_frame_with_charges(self, CalculatorClass, periodic, params):
        calculator = self.calculator(CalculatorClass, periodic, params)
        values = calculator.compute(*self.cscl_system_with_charges(periodic))
        assert_close(
            MADELUNG_CSCL,
            CHARGES_CSCL[0] * values[0, 0] + CHARGES_CSCL[1] * values[0, 1],
            atol=1e4,
            rtol=1e-5,
        )

    def test_multi_frame(self, CalculatorClass, periodic, params):
        calculator = self.calculator(CalculatorClass, periodic, params)
        if periodic:
            types, positions, cell = self.cscl_system(periodic)
            l_values = calculator.compute(
                types=[types, types],
                positions=[positions, positions],
                cell=[cell, cell],
            )
        else:
            types, positions = self.cscl_system(periodic)
            l_values = calculator.compute(
                types=[types, types], positions=[positions, positions]
            )

        for values in l_values:
            assert_close(
                MADELUNG_CSCL,
                CHARGES_CSCL[0] * values[0, 0] + CHARGES_CSCL[1] * values[0, 1],
                atol=1e4,
                rtol=1e-5,
            )

    def test_dtype_device(self, CalculatorClass, periodic, params):
        """Test that the output dtype and device are the same as the input."""
        device = "cpu"
        dtype = torch.float64

        types = torch.tensor([1], dtype=dtype, device=device)
        positions = torch.tensor([[0.0, 0.0, 0.0]], dtype=dtype, device=device)

        calculator = self.calculator(CalculatorClass, periodic, params)
        if periodic:
            cell = torch.eye(3, dtype=dtype, device=device)
            potential = calculator.compute(types=types, positions=positions, cell=cell)
        else:
            potential = calculator.compute(types=types, positions=positions)

        assert potential.dtype == dtype
        assert potential.device.type == device

    # Make sure that the calculators are computing the features without raising errors,
    # and returns the correct output format (TensorMap)
    def check_operation(self, CalculatorClass, periodic, params):
        calculator = self.calculator(CalculatorClass, periodic, params)

        if periodic:
            types, positions, cell = self.cscl_system(periodic)
            descriptor = calculator.compute(types=types, positions=positions, cell=cell)
        else:
            types, positions = self.cscl_system(periodic)
            descriptor = calculator.compute(types=types, positions=positions)

        assert type(descriptor) is torch.Tensor

    # Run the above test as a normal python script
    def test_operation_as_python(self, CalculatorClass, periodic, params):
        self.check_operation(CalculatorClass, periodic, params)

    # Similar to the above, but also testing that the code can be compiled as a torch
    # script
    # def test_operation_as_torch_script(self, CalculatorClass, periodic, params):
    #     scripted = torch.jit.script(CalculatorClass, periodic, params)
    #     self.check_operation(scripted)
