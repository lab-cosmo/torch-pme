from typing import Dict, List, Optional, Union

import torch


try:
    from metatensor.torch import Labels, TensorBlock, TensorMap
    from metatensor.torch.atomistic import System
except ImportError:
    raise ImportError(
        "metatensor.torch is required for meshlode.metatensor but is not installed. "
        "Try installing it with:\npip install metatensor[torch]"
    )

from ..calculators.base import CalculatorBase, _1d_tolist
from ..calculators.directpotential import _DirectPotentialImpl
from ..calculators.ewaldpotential import _EwaldPotentialImpl
from ..calculators.pmepotential import _PMEPotentialImpl


# We are breaking the Liskov substitution principle here by changing the signature of
# "compute" method to the supertype of metatansor class.
# mypy: disable-error-code="override"


class CalculatorBaseMetatensor(CalculatorBase):
    def __init__(self, exponent: float):
        super().__init__(exponent)

    def forward(self, systems: Union[List[System], System]) -> TensorMap:
        """Forward just calls :py:meth:`compute`."""
        return self.compute(systems)

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
        # Make sure that the compute function also works if only a single frame is
        # provided as input (for convenience of users testing out the code)
        if not isinstance(systems, list):
            systems = [systems]

        for system in systems:
            if system.dtype != systems[0].dtype:
                raise ValueError(
                    "`dtype` of all systems must be the same, got "
                    f"{system.dtype} and {systems[0].dtype}`"
                )

            if system.device != systems[0].device:
                raise ValueError(
                    "`device` of all systems must be the same, got "
                    f"{system.device} and {systems[0].device}`"
                )

        device = systems[0].positions.device

        all_atomic_types = torch.hstack([system.types for system in systems])
        atomic_types = _1d_tolist(torch.unique(all_atomic_types))
        n_types = len(atomic_types)

        has_charges = torch.tensor(["charges" in s.known_data() for s in systems])

        if not torch.all(has_charges):
            raise ValueError("`systems` do not consistently contain `charges` data")

        n_charges_channels = systems[0].get_data("charges").values.shape[1]
        spec_channels = list(range(n_charges_channels))
        key_names = ["center_type", "charges_channel"]

        for i_system, system in enumerate(systems):
            n_channels = system.get_data("charges").values.shape[1]
            if n_channels != n_charges_channels:
                raise ValueError(
                    f"number of charges-channels in system index {i_system} "
                    f"({n_channels}) is inconsistent with first system "
                    f"({n_charges_channels})"
                )

        # Initialize dictionary for TensorBlock storage.
        #
        # blocks are stored according to the `center_type` and `charge_channel`
        n_blocks = n_types * n_charges_channels
        feat_dic: Dict[int, List[torch.Tensor]] = {a: [] for a in range(n_blocks)}

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
                potential = self._compute_single_system(
                    positions=system.positions,
                    cell=system.cell,
                    charges=charges,
                    neighbor_indices=None,
                    neighbor_shifts=None,
                )
            else:
                potential = self._compute_single_system(
                    positions=system.positions,
                    charges=charges,
                    cell=system.cell,
                    neighbor_indices=neighbor_indices,
                    neighbor_shifts=neighbor_shifts,
                )

            # Reorder data into metatensor format
            for spec_center, at_num_center in enumerate(atomic_types):
                for spec_channel in range(len(spec_channels)):
                    a_pair = spec_center * n_charges_channels + spec_channel
                    feat_dic[a_pair] += [
                        potential[system.types == at_num_center, spec_channel]
                    ]

        # Assemble all computed potential values into TensorBlocks for each combination
        # of center_type and neighbor_type/charge_channel
        blocks: List[TensorBlock] = []
        for keys, values in feat_dic.items():
            spec_center = atomic_types[keys // n_charges_channels]

            # Generate the Labels objects for the samples and properties of the
            # TensorBlock.
            values_samples: List[List[int]] = []
            for i_frame, system in enumerate(systems):
                for i_atom in range(len(system)):
                    if system.types[i_atom] == spec_center:
                        values_samples.append([i_frame, i_atom])

            samples_vals_tensor = torch.tensor(
                values_samples, dtype=torch.int32, device=device
            )

            # If no atoms are found that match the types pair `samples_vals_tensor`
            # will be empty. We have to reshape the empty tensor to be a valid input for
            # `Labels`.
            if len(samples_vals_tensor) == 0:
                samples_vals_tensor = samples_vals_tensor.reshape(-1, 2)

            labels_samples = Labels(["system", "atom"], samples_vals_tensor)
            labels_properties = Labels(
                ["potential"], torch.tensor([[0]], device=device)
            )

            block = TensorBlock(
                samples=labels_samples,
                components=[],
                properties=labels_properties,
                values=torch.hstack(values).reshape((-1, 1)),
            )

            blocks.append(block)

        assert len(blocks) == n_blocks

        # Generate TensorMap from TensorBlocks by defining suitable keys
        key_values: List[torch.Tensor] = []
        for spec_center in atomic_types:
            for spec_channel in spec_channels:
                key_values.append(
                    torch.tensor([spec_center, spec_channel], device=device)
                )
        key_values = torch.vstack(key_values)

        labels_keys = Labels(key_names, key_values)

        return TensorMap(keys=labels_keys, blocks=blocks)


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
