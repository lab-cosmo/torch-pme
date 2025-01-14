"""
Basic tests if the calculator works and is torch scriptable. Actual tests are done
for the metatensor calculator.
"""

import io
import math

import pytest
import torch

from torchpme import (
    Calculator,
    CoulombPotential,
    EwaldCalculator,
    P3MCalculator,
    PMECalculator,
)

AVAILABLE_DEVICES = ["cpu"] + torch.cuda.is_available() * ["cuda"]
MADELUNG_CSCL = torch.tensor(2 * 1.7626 / math.sqrt(3))
CHARGES_CSCL = torch.tensor([1.0, -1.0])
SMEARING = 0.1
LR_WAVELENGTH = SMEARING / 4
MESH_SPACING = SMEARING / 4


@pytest.mark.parametrize("device", AVAILABLE_DEVICES)
@pytest.mark.parametrize(
    ("CalculatorClass", "params"),
    [
        (
            Calculator,
            {
                "potential": CoulombPotential(smearing=None),
            },
        ),
        (
            EwaldCalculator,
            {
                "potential": CoulombPotential(smearing=SMEARING),
                "lr_wavelength": LR_WAVELENGTH,
            },
        ),
        (
            PMECalculator,
            {
                "potential": CoulombPotential(smearing=SMEARING),
                "mesh_spacing": MESH_SPACING,
            },
        ),
        (
            P3MCalculator,
            {
                "potential": CoulombPotential(smearing=SMEARING),
                "mesh_spacing": MESH_SPACING,
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

    def test_smearing_non_positive(self, CalculatorClass, params, device):
        params = params.copy()
        match = r"`smearing` .* has to be positive"
        if type(CalculatorClass) in [EwaldCalculator, PMECalculator]:
            params["smearing"] = 0
            with pytest.raises(ValueError, match=match):
                CalculatorClass(**params, device=device)
            params["smearing"] = -0.1
            with pytest.raises(ValueError, match=match):
                CalculatorClass(**params, device=device)

    def test_interpolation_order_error(self, CalculatorClass, params, device):
        params = params.copy()
        if type(CalculatorClass) in [PMECalculator]:
            match = "Only `interpolation_nodes` from 1 to 5"
            params["interpolation_nodes"] = 10
            with pytest.raises(ValueError, match=match):
                CalculatorClass(**params, device=device)

    def test_lr_wavelength_non_positive(self, CalculatorClass, params, device):
        params = params.copy()
        match = r"`lr_wavelength` .* has to be positive"
        if type(CalculatorClass) in [EwaldCalculator]:
            params["lr_wavelength"] = 0
            with pytest.raises(ValueError, match=match):
                CalculatorClass(**params, device=device)
            params["lr_wavelength"] = -0.1
            with pytest.raises(ValueError, match=match):
                CalculatorClass(**params, device=device)

    def test_dtype_device(self, CalculatorClass, params, device):
        """Test that the output dtype and device are the same as the input."""
        dtype = torch.float64
        params = params.copy()
        positions = torch.tensor([[0.0, 0.0, 0.0]], dtype=dtype, device=device)
        charges = torch.ones((1, 2), dtype=dtype, device=device)
        cell = torch.eye(3, dtype=dtype, device=device)
        neighbor_indices = torch.tensor([[0, 0]], device=device)
        neighbor_distances = torch.tensor([0.1], device=device)
        params["potential"].device = device
        calculator = CalculatorClass(**params, device=device)
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

    def test_operation_as_python(self, CalculatorClass, params, device):
        """Run `check_operation` as a normal python script"""
        params = params.copy()
        params["potential"].device = device
        calculator = CalculatorClass(**params, device=device)
        self.check_operation(calculator=calculator, device=device)

    def test_operation_as_torch_script(self, CalculatorClass, params, device):
        """Run `check_operation` as a compiled torch script module."""
        params = params.copy()
        params["potential"].device = device
        calculator = CalculatorClass(**params, device=device)
        scripted = torch.jit.script(calculator)
        self.check_operation(calculator=scripted, device=device)

    def test_save_load(self, CalculatorClass, params, device):
        params = params.copy()
        params["potential"].device = device
        calculator = CalculatorClass(**params, device=device)
        scripted = torch.jit.script(calculator)
        with io.BytesIO() as buffer:
            torch.jit.save(scripted, buffer)
            buffer.seek(0)
            torch.jit.load(buffer)

    def test_prefactor(self, CalculatorClass, params, device):
        """Test if the prefactor is applied correctly."""
        params = params.copy()
        params["potential"].device = device
        prefactor = 2.0
        calculator1 = CalculatorClass(**params, device=device)
        calculator2 = CalculatorClass(**params, prefactor=prefactor, device=device)
        potentials1 = calculator1.forward(*self.cscl_system())
        potentials2 = calculator2.forward(*self.cscl_system())
        assert torch.allclose(potentials1 * prefactor, potentials2)

    def test_not_nan(self, CalculatorClass, params, device):
        """Make sure derivatives are not NaN."""
        params = params.copy()
        params["potential"].device = device

        calculator = CalculatorClass(**params, device=device)
        system = self.cscl_system(device)
        system[0].requires_grad = True
        system[1].requires_grad = True
        system[2].requires_grad = True
        system[-1].requires_grad = True
        energy = calculator.forward(*system).sum()

        # charges
        assert not torch.isnan(
            torch.autograd.grad(energy, system[0], retain_graph=True)[0]
        ).any()

        # neighbor distances
        assert not torch.isnan(
            torch.autograd.grad(energy, system[-1], retain_graph=True)[0]
        ).any()

        # positions, cell
        if CalculatorClass in [PMECalculator, P3MCalculator]:
            assert not torch.isnan(
                torch.autograd.grad(energy, system[1], retain_graph=True)[0]
            ).any()
            assert not torch.isnan(
                torch.autograd.grad(energy, system[2], retain_graph=True)[0]
            ).any()

    def test_dtype_and_device_incompatability(self, CalculatorClass, params, device):
        """Test that the calculator raises an error if the dtype and device are incompatible."""
        params = params.copy()
        params["potential"].device = device
        params["potential"].dtype = torch.float64
        with pytest.raises(AssertionError, match=".*dtype.*"):
            CalculatorClass(**params, dtype=torch.float32, device=device)
        with pytest.raises(AssertionError, match=".*device.*"):
            CalculatorClass(
                **params, dtype=params["potential"].dtype, device=torch.device("meta")
            )
