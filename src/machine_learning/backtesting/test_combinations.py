"""Tests for forecast combinations and their use in the harness."""

import numpy as np
import pandas as pd
import pytest

from backtesting.combinations import COMBINATIONS, LongHorizonBlend, add_combinations
from backtesting.data_prep import build_actuals_grid, load_orders
from backtesting.lightgbm_model import LGBM_VARIANTS
from backtesting.run_backtest import parse_args, resolve_models, run_fold
from backtesting.synthetic_data import generate_synthetic_orders

CUTOFF = pd.Timestamp("2025-01-31")
HORIZON = 40


def _series(values):
    dates = pd.date_range(CUTOFF + pd.Timedelta(days=1), periods=len(values))
    return pd.Series(np.asarray(values, dtype=float), index=dates)


def test_long_horizon_blend_leaves_early_days_alone():
    blend = LongHorizonBlend("m", "b", start=3, weight=0.25)
    out = blend.combine(_series([4, 4, 4, 4]), _series([0, 0, 8, 8]))
    np.testing.assert_allclose(out, [4, 4, 5, 5])


def test_add_combinations_uses_both_inputs():
    forecasts = {
        "lgbm_direct": {"a": _series([2.0] * 35)},
        "weekday_mean_4w": {"a": _series([4.0] * 35)},
    }
    out = add_combinations(forecasts, COMBINATIONS)
    blended = out["lgbm_direct_wm4w_long"]["a"]
    np.testing.assert_allclose(blended[:30], 2.0)
    np.testing.assert_allclose(blended[30:], 3.0)


def test_registered_level_adaptation_variants():
    assert LGBM_VARIANTS["lgbm_direct_train180"].train_days == 180
    assert LGBM_VARIANTS["lgbm_direct_scaled_28"].target_scale == 28
    assert resolve_models("lgbm_direct_wm4w_long") == ["lgbm_direct_wm4w_long"]
    assert COMBINATIONS["lgbm_direct_wm4w_long"].inputs == (
        "lgbm_direct",
        "weekday_mean_4w",
    )


@pytest.fixture(scope="module")
def orders():
    raw, _ = generate_synthetic_orders(
        n_skus=20, start="2024-01-01", end="2025-03-31", seed=5
    )
    return load_orders(raw)


def test_run_fold_scores_only_requested_models(orders, monkeypatch):
    blend = LongHorizonBlend("seasonal_naive_7", "moving_average_28")
    monkeypatch.setattr(
        "backtesting.run_backtest.COMBINATIONS", {"naive_ma_long": blend}
    )
    actuals = build_actuals_grid(orders)
    actuals_by_mlb = {m: f.set_index("date")["y"] for m, f in actuals.groupby("mlb")}
    rows = run_fold(
        orders,
        actuals_by_mlb,
        CUTOFF,
        HORIZON,
        max_series=5,
        models=["naive_ma_long", "moving_average_28"],
    )
    assert set(rows["model"]) == {"naive_ma_long", "moving_average_28"}
    for mlb, frame in rows.groupby("mlb"):
        wide = frame.pivot(index="h", columns="model", values="forecast")
        history = actuals_by_mlb[mlb][:CUTOFF]
        naive = np.resize(history.iloc[-7:].to_numpy(dtype=float), HORIZON)
        expected = np.where(
            wide.index >= 31, 0.5 * naive + 0.5 * wide["moving_average_28"], naive
        )
        np.testing.assert_allclose(wide["naive_ma_long"], expected)


def test_score_without_defaults_to_none():
    assert parse_args([]).score_without is None
    assert parse_args(["--score-without", "AWR1-110X74"]).score_without == (
        "AWR1-110X74"
    )
