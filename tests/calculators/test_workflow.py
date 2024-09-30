"""Basic tests if the calculator works and is torch scriptable. Actual tests are done
for the metatensor calculator."""

import io
import math

import pytest
import torch
from torch.testing import assert_close

from torchpme import DirectPotential, EwaldPotential, PMEPotential

AVAILABLE_DEVICES = [torch.device("cpu")] + torch.cuda.is_available() * [
    torch.device("cuda")
]
MADELUNG_CSCL = torch.tensor(2 * 1.7626 / math.sqrt(3))
CHARGES_CSCL = torch.tensor([1.0, -1.0])
ATOMIC_SMEARING = 0.1
LR_WAVELENGTH = ATOMIC_SMEARING / 4
MESH_SPACING = ATOMIC_SMEARING / 4
INTERPOLATION_ORDER = 2


@pytest.mark.parametrize(
    "CalculatorClass, params",
    [
        (DirectPotential, {}),
        (
            EwaldPotential,
            {
                "atomic_smearing": ATOMIC_SMEARING,
                "lr_wavelength": LR_WAVELENGTH,
            },
        ),
        (
            PMEPotential,
            {
                "atomic_smearing": ATOMIC_SMEARING,
                "mesh_spacing": MESH_SPACING,
                "interpolation_order": INTERPOLATION_ORDER,
            },
        ),
    ],
)
class TestWorkflow:
    def cscl_system(self, device=None):
        """CsCl crystal. Same as in the madelung test"""
        if device is None:
            device = torch.device("cpu")

        positions = torch.tensor([[0, 0, 0], [0.5, 0.5, 0.5]])
        charges = torch.tensor([1.0, -1.0]).reshape((-1, 1))
        cell = torch.eye(3)
        neighbor_indices = torch.tensor([[0, 1]], dtype=torch.int64)
        neighbor_distances = torch.tensor([0.8660])

        return (
            positions.to(device=device),
            charges.to(device=device),
            cell.to(device=device),
            neighbor_indices.to(device=device),
            neighbor_distances.to(device=device),
        )

    def test_atomic_smearing_non_positive(self, CalculatorClass, params):
        params = params.copy()
        match = r"`atomic_smearing` .* has to be positive"
        if type(CalculatorClass) in [EwaldPotential, PMEPotential]:
            params["atomic_smearing"] = 0
            with pytest.raises(ValueError, match=match):
                CalculatorClass(**params)
            params["atomic_smearing"] = -0.1
            with pytest.raises(ValueError, match=match):
                CalculatorClass(**params)

    def test_interpolation_order_error(self, CalculatorClass, params):
        params = params.copy()
        if type(CalculatorClass) in [PMEPotential]:
            match = "Only `interpolation_order` from 1 to 5"
            params["interpolation_order"] = 10
            with pytest.raises(ValueError, match=match):
                CalculatorClass(**params)

    def test_lr_wavelength_non_positive(self, CalculatorClass, params):
        params = params.copy()
        match = r"`lr_wavelength` .* has to be positive"
        if type(CalculatorClass) in [EwaldPotential]:
            params["lr_wavelength"] = 0
            with pytest.raises(ValueError, match=match):
                CalculatorClass(**params)
            params["lr_wavelength"] = -0.1
            with pytest.raises(ValueError, match=match):
                CalculatorClass(**params)

    def test_multi_frame(self, CalculatorClass, params):
        calculator = CalculatorClass(**params)
        positions, charges, cell, neighbor_indices, neighbor_distance = (
            self.cscl_system()
        )

        l_values = calculator.forward(
            positions=[positions, positions],
            cell=[cell, cell],
            charges=[charges, charges],
            neighbor_indices=[neighbor_indices, neighbor_indices],
            neighbor_distances=[neighbor_distance, neighbor_distance],
        )

        for values in l_values:
            assert_close(
                MADELUNG_CSCL,
                -torch.sum(charges * values),
                atol=1,
                rtol=1e-5,
            )

    def test_dtype_device(self, CalculatorClass, params):
        """Test that the output dtype and device are the same as the input."""
        device = "cpu"
        dtype = torch.float64

        positions = torch.tensor([[0.0, 0.0, 0.0]], dtype=dtype, device=device)
        charges = torch.ones((1, 2), dtype=dtype, device=device)
        cell = torch.eye(3, dtype=dtype, device=device)
        neighbor_indices = torch.tensor([[0, 0]], device=device)
        neighbor_distances = torch.tensor([0.1], device=device)

        calculator = CalculatorClass(**params)

        potential = calculator.forward(
            positions=positions,
            charges=charges,
            cell=cell,
            neighbor_indices=neighbor_indices,
            neighbor_distances=neighbor_distances,
        )

        assert potential.dtype == dtype
        assert potential.device.type == device

    def check_operation(self, calculator, device):
        """Make sure computation runs and returns a torch.Tensor."""
        descriptor = calculator.forward(*self.cscl_system(device))
        assert type(descriptor) is torch.Tensor

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
