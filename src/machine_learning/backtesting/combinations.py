"""
Forecasts built by combining other models' forecasts after they are made.

A combination names its input models; the harness computes them and keeps only the
models that were asked for.
"""

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LongHorizonBlend:
    """The model's forecast, averaged with a baseline from horizon day ``start`` on."""

    model: str
    baseline: str
    start: int = 31
    weight: float = 0.5  # Baseline weight from ``start`` on

    @property
    def inputs(self) -> Tuple[str, str]:
        return self.model, self.baseline

    def combine(self, model: pd.Series, baseline: pd.Series) -> pd.Series:
        """Blend two forecasts indexed by the same horizon dates."""
        steps = np.arange(1, len(model) + 1)
        weight = np.where(steps >= self.start, self.weight, 0.0)
        return (1 - weight) * model + weight * baseline.reindex(model.index)


COMBINATIONS: Dict[str, LongHorizonBlend] = {
    "lgbm_direct_wm4w_long": LongHorizonBlend("lgbm_direct", "weekday_mean_4w"),
}


def add_combinations(
    forecasts: Dict[str, Dict[str, pd.Series]], specs: Dict[str, LongHorizonBlend]
) -> Dict[str, Dict[str, pd.Series]]:
    """Add combinations (name -> spec) to ``forecasts`` (model -> series -> forecast)."""
    for name, spec in specs.items():
        model, baseline = (forecasts[n] for n in spec.inputs)
        forecasts[name] = {
            mlb: spec.combine(model[mlb], baseline[mlb])
            for mlb in model
            if mlb in baseline
        }
    return forecasts
