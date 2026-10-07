"""Backtest-only LightGBM variants and calibration on the shared forecasting core."""

from dataclasses import replace
from typing import Dict

import numpy as np
import pandas as pd

from backtesting.data_prep import build_training_features, eligible_series
from estimation import lightgbm_model as core
from estimation.lightgbm_model import (
    FEATURE_GROUPS,
    DEFAULT_CONFIG,
    LGBMConfig,
    LightGBMFold as _CoreLightGBMFold,
    horizon_dates,
)

# Retain the public backtest API while the implementations live in estimation.
_trailing_mean = core._trailing_mean
_direct_frame = core._direct_frame
_with_category = core._with_category
black_friday = core.black_friday
build_panel = core.build_panel
calendar_features = core.calendar_features
feature_columns = core.feature_columns
flag_stockouts = core.flag_stockouts


class LightGBMFold(_CoreLightGBMFold):
    """Shared model with backtest-only calibration methods."""

    def _validation_forecast(self, config: LGBMConfig):
        """Fit and select series at c - horizon; score their next horizon days."""
        inner_cutoff = self.cutoff - pd.Timedelta(days=self.horizon)
        key = (
            (
                replace(
                    config,
                    calibration_shrink=1.0,
                    calibration_floor=0.8,
                    constant_factor=1.0,
                ),
                inner_cutoff,
            )
            if not config.params
            else None
        )
        if key is not None and key in self._validation:
            return self._validation[key]
        inner = LightGBMFold(self.orders, inner_cutoff, self.horizon)
        if self._inner_eligible is None:
            features = build_training_features(self.orders, inner_cutoff)
            self._inner_eligible = eligible_series(features, inner_cutoff, self.horizon)
        names = self._inner_eligible
        self.validation_series = names
        inner_index = {name: i for i, name in enumerate(inner.panel.mlbs)}
        outer_index = {name: i for i, name in enumerate(self.panel.mlbs)}
        inner_rows = np.array([inner_index[name] for name in names], dtype=int)
        outer_rows = np.array([outer_index[name] for name in names], dtype=int)
        if not len(inner_rows):
            return (
                inner_rows,
                np.empty((0, self.horizon)),
                np.empty((0, self.horizon)),
                np.array([]),
            )
        pred = inner._forecast_direct(replace(config, calibration="none"), inner_rows)
        dates = horizon_dates(inner_cutoff, self.horizon)
        actual = self.panel.sales[outer_rows][:, self.panel.dates.get_indexer(dates)]
        actual = np.nan_to_num(actual, nan=0.0)
        level = inner.matrices(config.stockout).rmean_28[inner_rows, -1]
        result = (inner_rows, pred, actual, level)
        if key is not None:
            self._validation[key] = result
        return result

    @staticmethod
    def _tiers(level: np.ndarray) -> np.ndarray:
        """Top 20% by origin level, matching the harness's tier share."""
        top = np.zeros(len(level), dtype=bool)
        if len(level):
            top[
                np.argsort(-np.nan_to_num(level))[: max(1, round(len(level) * 0.2))]
            ] = True
        return top

    def _forecast_calibrated(self, config: LGBMConfig, rows: np.ndarray) -> np.ndarray:
        self.calibration_factors = {}
        self.calibration_diagnostics = []
        if config.calibration not in {
            "overall",
            "block",
            "tier",
            "block_tier",
            "power",
            "constant",
        }:
            raise ValueError(f"Unknown calibration: {config.calibration}")
        if config.calibration == "constant":
            factor = config.constant_factor
            self.calibration_factors[((1, self.horizon), None)] = factor
            self.calibration_diagnostics.append(
                {
                    "block_start": 1,
                    "block_end": self.horizon,
                    "tier": "all",
                    "validation_series": 0,
                    "validation_actual": np.nan,
                    "validation_forecast": np.nan,
                    "raw_factor": factor,
                    "applied_factor": factor,
                    "floor_binds": False,
                }
            )
            return self._forecast_direct(config, rows) * factor
        chosen = config
        if config.calibration == "power":
            scored = []
            for power in (1.1, 1.2, 1.3, 1.4, 1.5):
                candidate = replace(config, tweedie_power=power, calibration="none")
                _, pred, actual, _ = self._validation_forecast(candidate)
                total = actual.sum()
                if total > 0:
                    bias = (pred.sum() - total) / total
                    wape = np.abs(pred - actual).sum() / total
                    scored.append((abs(bias) > 0.05, wape, abs(bias), power))
            if scored:
                eligible = [s for s in scored if not s[0]]
                # If no power meets the constraint, choose the least biased one.
                self.selected_power = (
                    min(eligible, key=lambda s: s[1])[3]
                    if eligible
                    else min(scored, key=lambda s: s[2])[3]
                )
                chosen = replace(config, tweedie_power=self.selected_power)
            return self._forecast_direct(chosen, rows)

        valid, pred, actual, level = self._validation_forecast(
            replace(config, calibration="none")
        )
        result = self._forecast_direct(config, rows)
        if not len(valid):
            return result
        use_block = config.calibration in {"block", "block_tier"}
        use_tier = config.calibration in {"tier", "block_tier"}
        validation_tier = self._tiers(level)
        current_level = self.matrices(config.stockout).rmean_28[rows, -1]
        current_tier = self._tiers(current_level)
        for block in config.blocks if use_block else ((1, self.horizon),):
            start, end = block[0] - 1, min(block[1], self.horizon)
            if start >= end:
                continue
            for tier in (False, True) if use_tier else (None,):
                mask = (
                    validation_tier == tier
                    if use_tier
                    else np.ones(len(valid), dtype=bool)
                )
                observed = actual[mask, start:end].sum()
                forecast = pred[mask, start:end].sum()
                # A minimum volume prevents a handful of intermittent days from
                # setting a noisy ratio. Shrink and bound the remaining ratios.
                raw_factor = (
                    np.clip(observed / forecast, 0.8, 1.5) if forecast >= 50 else 1.0
                )
                raw_factor = float(raw_factor) if np.isfinite(raw_factor) else 1.0
                factor = float(
                    np.clip(
                        max(
                            config.calibration_floor,
                            1 + config.calibration_shrink * (raw_factor - 1),
                        ),
                        0.8,
                        1.5,
                    )
                )
                self.calibration_factors[(block, tier)] = factor
                self.calibration_diagnostics.append(
                    {
                        "block_start": block[0],
                        "block_end": min(block[1], self.horizon),
                        "tier": ("top 20%" if tier else "rest") if use_tier else "all",
                        "validation_series": int(mask.sum()),
                        "validation_actual": float(observed),
                        "validation_forecast": float(forecast),
                        "raw_factor": raw_factor,
                        "applied_factor": factor,
                        "floor_binds": 1 + config.calibration_shrink * (raw_factor - 1)
                        < config.calibration_floor,
                    }
                )
                selected = (
                    current_tier == tier if use_tier else np.ones(len(rows), dtype=bool)
                )
                result[selected, start:end] *= factor
        return result


def forecast_lightgbm(orders, cutoff, horizon, mlbs, configs):
    fold = LightGBMFold(orders, cutoff, horizon)
    return {name: fold.forecast(config, mlbs) for name, config in configs.items()}


UNCALIBRATED_CONFIG = DEFAULT_CONFIG
SHRUNK_CONFIG = replace(
    UNCALIBRATED_CONFIG,
    calibration="block_tier",
    calibration_shrink=0.35,
    calibration_floor=1.06,
)


def _variants() -> Dict[str, LGBMConfig]:
    """
    Named variants: the default direct and recursive models plus ablations.

    For each strategy: feature groups added in turn (``_fs_<groups>``), the default
    with one element removed or added (``_no_id``, ``_no_lag364``, ``_no_stockout``,
    ``_with_month``), and Tweedie variance powers 1.1-1.5 (``_p<power>``).
    """
    variants = {}
    for strategy in ("direct", "recursive"):
        base = replace(UNCALIBRATED_CONFIG, strategy=strategy)
        prefix = f"lgbm_{strategy}"
        variants[prefix] = base
        for n in range(1, len(FEATURE_GROUPS)):
            groups = FEATURE_GROUPS[:n]
            variants[f"{prefix}_fs_{'+'.join(groups)}"] = replace(
                base, feature_groups=groups
            )
        variants[f"{prefix}_no_id"] = replace(base, use_series_id=False)
        variants[f"{prefix}_no_lag364"] = replace(base, use_lag_364=False)
        variants[f"{prefix}_no_stockout"] = replace(base, stockout="none")
        variants[f"{prefix}_with_month"] = replace(base, drop_columns=())
        for power in (1.1, 1.2, 1.3, 1.4, 1.5):
            variants[f"{prefix}_p{power:.1f}"] = replace(base, tweedie_power=power)
    for method in ("overall", "block", "tier", "block_tier", "power"):
        variants[f"lgbm_direct_cal_{method}"] = replace(
            UNCALIBRATED_CONFIG, calibration=method
        )
    for window in (28, 91):
        variants[f"lgbm_direct_scaled_{window}"] = replace(
            UNCALIBRATED_CONFIG, target_scale=window
        )
    variants["lgbm_direct_uncalibrated"] = UNCALIBRATED_CONFIG
    variants["lgbm_direct_cal_shrunk"] = SHRUNK_CONFIG
    variants["lgbm_direct_constant_1p10"] = replace(
        UNCALIBRATED_CONFIG, calibration="constant", constant_factor=1.10
    )
    variants["lgbm_blend_50_50"] = replace(UNCALIBRATED_CONFIG, strategy="blend")
    variants["lgbm_direct"] = DEFAULT_CONFIG
    return variants


LGBM_VARIANTS: Dict[str, LGBMConfig] = _variants()
