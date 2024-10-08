"""Basic tests if the calculator works and is torch scriptable. Actual tests are done
for the metatensor calculator."""

import io
import math

import pytest
import torch

from torchpme import Calculator, CoulombPotential, EwaldCalculator, PMECalculator

AVAILABLE_DEVICES = [torch.device("cpu")] + torch.cuda.is_available() * [
    torch.device("cuda")
]
MADELUNG_CSCL = torch.tensor(2 * 1.7626 / math.sqrt(3))
CHARGES_CSCL = torch.tensor([1.0, -1.0])
RANGE_RADIUS = 0.1
LR_WAVELENGTH = RANGE_RADIUS / 4
MESH_SPACING = RANGE_RADIUS / 4
NUM_NODES_PER_AXIS = 3


@pytest.mark.parametrize(
    "CalculatorClass, params",
    [
        (
            Calculator,
            {
                "potential": CoulombPotential(range_radius=None),
            },
        ),
        (
            EwaldCalculator,
            {
                "potential": CoulombPotential(range_radius=RANGE_RADIUS),
                "lr_wavelength": LR_WAVELENGTH,
            },
        ),
        (
            PMECalculator,
            {
                "potential": CoulombPotential(range_radius=RANGE_RADIUS),
                "mesh_spacing": MESH_SPACING,
                "num_nodes_per_axis": NUM_NODES_PER_AXIS,
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
            charges.to(device=device),
            cell.to(device=device),
            positions.to(device=device),
            neighbor_indices.to(device=device),
            neighbor_distances.to(device=device),
        )

    def test_range_radius_non_positive(self, CalculatorClass, params):
        params = params.copy()
        match = r"`range_radius` .* has to be positive"
        if type(CalculatorClass) in [EwaldCalculator, PMECalculator]:
            params["range_radius"] = 0
            with pytest.raises(ValueError, match=match):
                CalculatorClass(**params)
            params["range_radius"] = -0.1
            with pytest.raises(ValueError, match=match):
                CalculatorClass(**params)

    def test_interpolation_order_error(self, CalculatorClass, params):
        params = params.copy()
        if type(CalculatorClass) in [PMECalculator]:
            match = "Only `num_nodes_per_axis` from 1 to 5"
            params["num_nodes_per_axis"] = 10
            with pytest.raises(ValueError, match=match):
                CalculatorClass(**params)

    def test_lr_wavelength_non_positive(self, CalculatorClass, params):
        params = params.copy()
        match = r"`lr_wavelength` .* has to be positive"
        if type(CalculatorClass) in [EwaldCalculator]:
            params["lr_wavelength"] = 0
            with pytest.raises(ValueError, match=match):
                CalculatorClass(**params)
            params["lr_wavelength"] = -0.1
            with pytest.raises(ValueError, match=match):
                CalculatorClass(**params)

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
            charges=charges,
            cell=cell,
            positions=positions,
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
