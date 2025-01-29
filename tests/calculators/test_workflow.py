"""
Basic tests if the calculator works and is torch scriptable. Actual tests are done
for the metatensor calculator.
"""

import io

import pytest
import torch

from torchpme import (
    Calculator,
    CoulombPotential,
    EwaldCalculator,
    P3MCalculator,
    PMECalculator,
)

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
    def cscl_system(self, device=None, dtype=None):
        """CsCl crystal. Same as in the madelung test"""
        device = torch.get_default_device() if device is None else torch.device(device)
        dtype = torch.get_default_dtype() if dtype is None else dtype

        positions = torch.tensor(
            [[0, 0, 0], [0.5, 0.5, 0.5]], dtype=dtype, device=device
        )
        charges = torch.tensor([1.0, -1.0], dtype=dtype, device=device).reshape((-1, 1))
        cell = torch.eye(3, dtype=dtype, device=device)
        neighbor_indices = torch.tensor([[0, 1]], dtype=torch.int64, device=device)
        neighbor_distances = torch.tensor([0.8660], dtype=dtype, device=device)

        return charges, cell, positions, neighbor_indices, neighbor_distances

    def test_smearing_non_positive(self, CalculatorClass, params, device, dtype):
        params = params.copy()
        match = r"`smearing` .* has to be positive"
        if type(CalculatorClass) in [EwaldCalculator, PMECalculator]:
            params["smearing"] = 0
            with pytest.raises(ValueError, match=match):
                CalculatorClass(**params, device=device, dtype=dtype)
            params["smearing"] = -0.1
            with pytest.raises(ValueError, match=match):
                CalculatorClass(**params, device=device, dtype=dtype)

    def test_interpolation_order_error(self, CalculatorClass, params, device, dtype):
        params = params.copy()
        if type(CalculatorClass) in [PMECalculator]:
            match = "Only `interpolation_nodes` from 1 to 5"
            params["interpolation_nodes"] = 10
            with pytest.raises(ValueError, match=match):
                CalculatorClass(**params, device=device, dtype=dtype)

    def test_lr_wavelength_non_positive(self, CalculatorClass, params, device, dtype):
        params = params.copy()
        match = r"`lr_wavelength` .* has to be positive"
        if type(CalculatorClass) in [EwaldCalculator]:
            params["lr_wavelength"] = 0
            with pytest.raises(ValueError, match=match):
                CalculatorClass(**params, device=device, dtype=dtype)
            params["lr_wavelength"] = -0.1
            with pytest.raises(ValueError, match=match):
                CalculatorClass(**params, device=device, dtype=dtype)

    def test_dtype_device(self, CalculatorClass, params, device, dtype):
        """Test that the output dtype and device are the same as the input."""
        params = params.copy()
        params["potential"].device = device
        params["potential"].dtype = dtype

        calculator = CalculatorClass(**params, device=device, dtype=dtype)
        potential = calculator.forward(*self.cscl_system(device=device, dtype=dtype))

        assert potential.dtype == dtype

        if isinstance(device, torch.device):
            assert potential.device == device
        else:
            assert potential.device.type == device

    def check_operation(self, calculator, device, dtype):
        """Make sure computation runs and returns a torch.Tensor."""
        descriptor = calculator.forward(*self.cscl_system(device=device, dtype=dtype))
        assert type(descriptor) is torch.Tensor

    def test_operation_as_python(self, CalculatorClass, params, device, dtype):
        """Run `check_operation` as a normal python script"""
        params = params.copy()
        params["potential"].device = device
        params["potential"].dtype = dtype

        calculator = CalculatorClass(**params, device=device, dtype=dtype)
        self.check_operation(calculator=calculator, device=device, dtype=dtype)

    def test_operation_as_torch_script(self, CalculatorClass, params, device, dtype):
        """Run `check_operation` as a compiled torch script module."""
        params = params.copy()
        params["potential"].device = device
        params["potential"].dtype = dtype

        calculator = CalculatorClass(**params, device=device, dtype=dtype)
        scripted = torch.jit.script(calculator)
        self.check_operation(calculator=scripted, device=device, dtype=dtype)

    def test_save_load(self, CalculatorClass, params, device, dtype):
        params = params.copy()
        params["potential"].device = device
        params["potential"].dtype = dtype

        calculator = CalculatorClass(**params, device=device, dtype=dtype)
        scripted = torch.jit.script(calculator)
        with io.BytesIO() as buffer:
            torch.jit.save(scripted, buffer)
            buffer.seek(0)
            torch.jit.load(buffer)

    def test_prefactor(self, CalculatorClass, params, device, dtype):
        """Test if the prefactor is applied correctly."""
        params = params.copy()
        params["potential"].device = device
        params["potential"].dtype = dtype

        prefactor = 2.0
        calculator1 = CalculatorClass(**params, device=device, dtype=dtype)
        calculator2 = CalculatorClass(
            **params, prefactor=prefactor, device=device, dtype=dtype
        )

        potentials1 = calculator1.forward(*self.cscl_system(device=device, dtype=dtype))
        potentials2 = calculator2.forward(*self.cscl_system(device=device, dtype=dtype))

        assert torch.allclose(potentials1 * prefactor, potentials2)

    def test_not_nan(self, CalculatorClass, params, device, dtype):
        """Make sure derivatives are not NaN."""
        params = params.copy()
        params["potential"].device = device
        params["potential"].dtype = dtype

        calculator = CalculatorClass(**params, device=device, dtype=dtype)
        system = self.cscl_system(device=device, dtype=dtype)
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

    def test_dtype_and_device_incompatability(
        self, CalculatorClass, params, device, dtype
    ):
        """Test that the calculator raises an error if the dtype and device are incompatible with potential."""
        params = params.copy()

        other_dtype = torch.float32 if dtype == torch.float64 else torch.float64

        params["potential"].device = device
        params["potential"].dtype = dtype

        match = (
            rf"dtype of `potential` \({params['potential'].dtype}\) must be same as "
            rf"of `calculator` \({other_dtype}\)"
        )
        with pytest.raises(TypeError, match=match):
            CalculatorClass(**params, dtype=other_dtype, device=device)

        match = (
            rf"device of `potential` \({params['potential'].device}\) must be same as "
            rf"of `calculator` \(meta\)"
        )
        with pytest.raises(ValueError, match=match):
            CalculatorClass(**params, dtype=dtype, device=torch.device("meta"))

    def test_potential_and_calculator_incompatability(
        self,
        CalculatorClass,
        params,
        device,
        dtype,
    ):
        """Test that the calculator raises an error if the potential and calculator are incompatible."""
        params = params.copy()
        params["potential"].device = device
        params["potential"].dtype = dtype

        params["potential"] = torch.jit.script(params["potential"])
        with pytest.raises(
            TypeError, match="Potential must be an instance of Potential, got.*"
        ):
            CalculatorClass(**params, device=device, dtype=dtype)
