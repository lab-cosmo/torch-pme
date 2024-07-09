from typing import List, Optional, Union

import torch


try:
    from metatensor.torch import Labels, TensorBlock, TensorMap
    from metatensor.torch.atomistic import System
except ImportError:
    raise ImportError(
        "metatensor.torch is required for meshlode.metatensor but is not installed. "
        "Try installing it with:\npip install metatensor[torch]"
    )

from ..calculators.base import CalculatorBase
from ..calculators.directpotential import _DirectPotentialImpl
from ..calculators.ewaldpotential import _EwaldPotentialImpl
from ..calculators.pmepotential import _PMEPotentialImpl

@torch.jit.script
def _1d_tolist(x: torch.Tensor) -> List[int]:
    """Auxilary function to convert 1d torch tensor to list of integers."""
    result: List[int] = []
    for i in x:
        result.append(i.item())
    return result


class CalculatorBaseMetatensor(CalculatorBase):
    def __init__(self, exponent: float):
        super().__init__(exponent)

    def forward(self, systems: Union[List[System], System]) -> TensorMap:
        """Forward just calls :py:meth:`compute`."""
        return self.compute(systems)

    def _validate_compute_parameters(
        self, systems: Union[List[System], System]
    ) -> List[System]:
        # Make sure that the compute function also works if only a single frame is
        # provided as input (for convenience of users testing out the code)
        if not isinstance(systems, list):
            systems = [systems]

        self._device = systems[0].positions.device
        for system in systems:
            if system.dtype != systems[0].dtype:
                raise ValueError(
                    "`dtype` of all systems must be the same, got "
                    f"{system.dtype} and {systems[0].dtype}`"
                )

            if system.device != self._device:
                raise ValueError(
                    "`device` of all systems must be the same, got "
                    f"{system.device} and {systems[0].device}`"
                )

        has_charges = torch.tensor(["charges" in s.known_data() for s in systems])
        if not torch.all(has_charges):
            raise ValueError("`systems` do not consistently contain `charges` data")

        self._n_charges_channels = systems[0].get_data("charges").values.shape[1]
        for i_system, system in enumerate(systems):
            n_channels = system.get_data("charges").values.shape[1]
            if n_channels != self._n_charges_channels:
                raise ValueError(
                    f"number of charges-channels in system index {i_system} "
                    f"({n_channels}) is inconsistent with first system "
                    f"({self._n_charges_channels})"
                )

        return systems

    def compute(self, systems: Union[List[System], System]) -> TensorMap:
        """Compute potential for all provided ``systems``.

        All ``systems`` must have the same ``dtype`` and the same ``device``. If each
        system contains a custom data field `charges` the potential will be calculated
        for each "charges-channel". The number of `charges-channels` must be same in all
        ``systems``. If no "explicit" charges are set the potential will be calculated
        for each "types-channels".

        Refer to :meth:`meshlode.PMEPotential.compute()` for additional details on how
        "charges-channel" and "types-channels" are computed.

        :param systems: single System or list of
            :py:class:`metatensor.torch.atomisic.System` on which to run the
            calculations.

        :return: TensorMap containing the potential of all types. The keys of the
            TensorMap are "center_type" and "neighbor_type" if no charges are asociated
            with
        """
        systems = self._validate_compute_parameters(systems)
        potentials: List[torch.Tensor] = []

        for system in systems:
            charges = system.get_data("charges").values

            # try to extract neighbor list from system object
            neighbor_indices = None
            for neighbor_list_options in system.known_neighbor_lists():
                if (
                    hasattr(self, "sr_cutoff")
                    and neighbor_list_options.cutoff == self.sr_cutoff
                ):
                    neighbor_list = system.get_neighbor_list(neighbor_list_options)

                    neighbor_indices = neighbor_list.samples.values[:, :2]
                    neighbor_shifts = neighbor_list.samples.values[:, 2:]

                    break

            if neighbor_indices is None:
                potentials.append(
                    self._compute_single_system(
                        positions=system.positions,
                        cell=system.cell,
                        charges=charges,
                        neighbor_indices=None,
                        neighbor_shifts=None,
                    )
                )
            else:
                potentials.append(
                    self._compute_single_system(
                        positions=system.positions,
                        charges=charges,
                        cell=system.cell,
                        neighbor_indices=neighbor_indices,
                        neighbor_shifts=neighbor_shifts,
                    )
                )

        values_samples: List[List[int]] = []
        for i_system in range(len(systems)):
            for i_atom in range(len(system)):
                values_samples.append([i_system, i_atom])

        samples_vals_tensor = torch.tensor(values_samples, device=self._device)

        block = TensorBlock(
            values=torch.vstack(potentials),
            samples=Labels(["system", "atom"], samples_vals_tensor),
            components=[],
            properties=Labels.range("charges_channel", self._n_charges_channels),
        )

        return TensorMap(keys=Labels.single(), blocks=[block])


class DirectPotential(CalculatorBaseMetatensor, _DirectPotentialImpl):
    """Specie-wise long-range potential using a direct summation over all atoms.

    Refer to :class:`meshlode.DirectPotential` for full documentation.
    """

    def __init__(self, exponent: float = 1.0):
        self._DirectPotentialImpl.__init__(self, exponent=exponent)
        CalculatorBaseMetatensor.__init__(self, exponent=exponent)


class EwaldPotential(CalculatorBaseMetatensor, _EwaldPotentialImpl):
    """Specie-wise long-range potential computed using the Ewald sum.

    Refer to :class:`meshlode.EwaldPotential` for full documentation.
    """

    def __init__(
        self,
        exponent: float = 1.0,
        sr_cutoff: Optional[torch.Tensor] = None,
        atomic_smearing: Optional[float] = None,
        lr_wavelength: Optional[float] = None,
        subtract_self: Optional[bool] = True,
        subtract_interior: Optional[bool] = False,
    ):
        _EwaldPotentialImpl.__init__(
            self,
            exponent=exponent,
            sr_cutoff=sr_cutoff,
            atomic_smearing=atomic_smearing,
            lr_wavelength=lr_wavelength,
            subtract_self=subtract_self,
            subtract_interior=subtract_interior,
        )
        CalculatorBaseMetatensor.__init__(self, exponent=exponent)


class PMEPotential(CalculatorBaseMetatensor, _PMEPotentialImpl):
    """Specie-wise long-range potential using a particle mesh-based Ewald (PME).

    Refer to :class:`meshlode.PMEPotential` for full documentation.
    """

    def __init__(
        self,
        exponent: float = 1.0,
        sr_cutoff: Optional[torch.Tensor] = None,
        atomic_smearing: Optional[float] = None,
        mesh_spacing: Optional[float] = None,
        interpolation_order: Optional[int] = 3,
        subtract_self: Optional[bool] = True,
        subtract_interior: Optional[bool] = False,
    ):
        _PMEPotentialImpl.__init__(
            self,
            exponent=exponent,
            sr_cutoff=sr_cutoff,
            atomic_smearing=atomic_smearing,
            mesh_spacing=mesh_spacing,
            interpolation_order=interpolation_order,
            subtract_self=subtract_self,
            subtract_interior=subtract_interior,
        )
        CalculatorBaseMetatensor.__init__(self, exponent=exponent)
