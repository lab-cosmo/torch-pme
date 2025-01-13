from ...calculators import EwaldCalculator, PMECalculator, P3MCalculator
from ...potentials import InversePowerLawPotential
from . import TunerBase
from .error_bounds import EwaldErrorBounds, PMEErrorBounds, P3MErrorBounds


class GridSearchTuner(TunerBase):
    def tune(self, accuracy: float = 1e-3):
        if self.calculator is EwaldCalculator:
            error_bounds = EwaldErrorBounds(self.charges, self.cell, self.positions)
        elif self.calculator is PMECalculator:
            error_bounds = PMEErrorBounds(self.charges, self.cell, self.positions)
        elif self.calculator is P3MCalculator:
            error_bounds = P3MErrorBounds(self.charges, self.cell, self.positions)
        else:
            raise NotImplementedError

        smearing = self.estimate_smearing(accuracy)
        param_errors = []
        param_timings = []
        for param in self.params:
            error = error_bounds(smearing=smearing, cutoff=self.cutoff, **param)
            param_errors.append(float(error))
            if error > accuracy:
                param_timings.append(float("inf"))
                continue

            param_timings.append(self._timing(smearing, param))

        return param_errors, param_timings

    def _timing(self, smearing: float, k_space_params: dict):
        calculator = self.calculator(
            potential=InversePowerLawPotential(
                exponent=self.exponent,  # but only exponent = 1 is supported
                smearing=smearing,
            ),
            **k_space_params,
        )

        return self.time_func(calculator)
