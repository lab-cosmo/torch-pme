import math
from typing import Optional

import torch

from . import (
    _estimate_smearing_cutoff,
    _optimize_parameters,
    _validate_parameters,
)

TWO_PI = 2 * math.pi


class Tuner(torch.nn.Module):
    def __init__(self, max_steps: int = 50000, learning_rate: float = 0.1):
        super().__init__()
        self.max_steps = max_steps
        self.learning_rate = learning_rate

    def forward(
        self,
        sum_squared_charges: float,
        cell: torch.Tensor,
        positions: torch.Tensor,
        smearing: Optional[float] = None,
        lr_wavelength: Optional[float] = None,
        cutoff: Optional[float] = None,
        exponent: int = 1,
        accuracy: float = 1e-3,
    ):
        _validate_parameters(sum_squared_charges, cell, positions, exponent, accuracy)

        params = self._init_params(
            cell=cell,
            smearing=smearing,
            lr_wavelength=lr_wavelength,
            cutoff=cutoff,
            accuracy=accuracy,
        )

        _optimize_parameters(
            params=params,
            loss=self.loss,
            max_steps=self.max_steps,
            accuracy=accuracy,
            learning_rate=self.learning_rate,
        )

        return self._post_process(params)

    def _init_params(self, cell, smearing, lr_wavelength, cutoff, accuracy):
        return _estimate_smearing_cutoff(
            cell=cell, smearing=smearing, cutoff=cutoff, accuracy=accuracy
        )
    
    def _post_process(self, params):
        smearing_opt, k_cutoff_opt, cutoff_opt = params
        return (
            float(smearing_opt),
            {"lr_wavelength": TWO_PI / float(k_cutoff_opt)},
            float(cutoff_opt),
        )