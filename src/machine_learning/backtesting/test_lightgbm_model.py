"""Tests for the global LightGBM challenger."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from backtesting.data_prep import build_actuals_grid, load_orders
from backtesting.lightgbm_model import (
    LGBM_VARIANTS,
    LGBMConfig,
    LightGBMFold,
    _trailing_mean,
    black_friday,
    build_panel,
    calendar_features,
    feature_columns,
    flag_stockouts,
)
from backtesting.run_backtest import resolve_models, run_fold
from backtesting.synthetic_data import generate_synthetic_orders

CUTOFF = pd.Timestamp("2025-01-31")
HORIZON = 14
FAST = {"n_estimators": 20, "train_days": 120}


@pytest.fixture(scope="module")
def orders():
    raw, _ = generate_synthetic_orders(
        n_skus=30, start="2024-01-01", end="2025-03-31", seed=5
    )
    return load_orders(raw)


@pytest.fixture(scope="module")
def fold(orders):
    return LightGBMFold(orders, CUTOFF, HORIZON)


def _busiest(orders, n=5):
    history = orders[orders["date"] <= CUTOFF]
    return history.groupby("mlb")["order_items_quantity"].sum().nlargest(n).index


def test_panel_stops_at_cutoff_and_derives_unit_price(orders):
    panel = build_panel(orders, CUTOFF)
    assert panel.dates[-1] == CUTOFF
    assert panel.sales.shape == panel.price.shape == (len(panel.mlbs), len(panel.dates))

    # Every series starts on its first sale; nothing is filled before it
    for row, first in zip(panel.sales, panel.first):
        assert np.isnan(row[:first]).all()
        assert row[first] > 0

    # Unit price is revenue / units on sale days
    mlb = panel.mlbs[0]
    day = panel.dates[panel.first[0]]
    sold = orders[(orders["mlb"] == mlb) & (orders["date"] == day)]
    expected = (
        sold["order_items_quantity"] * sold["order_items_unit_price"]
    ).sum() / sold["order_items_quantity"].sum()
    assert panel.price[0, panel.first[0]] == pytest.approx(expected)


@pytest.mark.parametrize("strategy", ["direct", "recursive"])
def test_forecasts_ignore_data_after_cutoff(orders, strategy):
    """Changing every order after the cutoff must not change any forecast."""
    config = replace(LGBMConfig(strategy=strategy), **FAST)
    mlbs = list(_busiest(orders))
    baseline = LightGBMFold(orders, CUTOFF, HORIZON).forecast(config, mlbs)

    tampered = orders.copy()
    after = tampered["date"] > CUTOFF
    tampered.loc[after, "order_items_quantity"] *= 50
    tampered.loc[after, "order_items_unit_price"] *= 3
    changed = LightGBMFold(tampered, CUTOFF, HORIZON).forecast(config, mlbs)

    for mlb in mlbs:
        pd.testing.assert_series_equal(baseline[mlb], changed[mlb])


@pytest.mark.parametrize("strategy", ["direct", "recursive"])
def test_forecast_shape_dates_and_non_negative(orders, fold, strategy):
    config = replace(LGBMConfig(strategy=strategy), **FAST)
    mlbs = list(_busiest(orders, 8))
    forecasts = fold.forecast(config, mlbs + ["NOT_A_SERIES"])

    assert set(forecasts) == set(mlbs)
    expected_dates = pd.date_range(CUTOFF + pd.Timedelta(days=1), periods=HORIZON)
    for forecast in forecasts.values():
        assert forecast.index.equals(expected_dates)
        assert forecast.notna().all()
        assert (forecast >= 0).all()


def test_training_frames_respect_blocks_and_stockout_mask(fold):
    """Training targets are built inside ``_*_training``, which asserts date <= cutoff."""
    config = replace(LGBMConfig(), **FAST)
    for block, step in zip(config.blocks, config.origin_steps):
        frame, y = fold._direct_training(config, block, step)
        assert len(frame) == len(y) > 0
        assert frame["h"].between(*block).all()
        assert (y >= 0).all()

    masked, _ = fold._recursive_training(config)
    unmasked, _ = fold._recursive_training(replace(config, stockout="none"))
    assert len(masked) < len(unmasked)  # Flagged stockout days are dropped as targets


def test_direct_blocks_only_use_lags_known_at_the_origin():
    config = LGBMConfig()
    assert "lag_7" in feature_columns(config, (1, 7))
    assert "lag_7" not in feature_columns(config, (8, 30))
    assert "lag_28" in feature_columns(config, (8, 28))
    assert not {"lag_14", "lag_28"} & set(feature_columns(config, (31, 90)))
    assert "lag_1" in feature_columns(replace(config, strategy="recursive"))


def test_feature_groups_and_drops():
    calendar_only = LGBMConfig(feature_groups=("calendar",))
    assert feature_columns(calendar_only, (1, 7)) == ["h", "dow", "dom"]
    with_month = LGBMConfig(feature_groups=("calendar",), drop_columns=())
    assert feature_columns(with_month, (1, 7)) == ["h", "dow", "dom", "month"]
    no_id = LGBMConfig(use_series_id=False, use_lag_364=False, drop_columns=("dom",))
    columns = feature_columns(no_id, (1, 7))
    assert not {"series", "lag_364", "dom"} & set(columns)


def test_trailing_mean_ignores_missing_values():
    values = np.array([[1.0, np.nan, 3.0, 5.0]])
    result = _trailing_mean(values, 2)
    np.testing.assert_allclose(result, [[1.0, 1.0, 3.0, 4.0]])


def test_flag_stockouts_targets_implausible_closed_runs():
    busy = np.r_[np.full(28, 2.0), np.zeros(5), np.full(5, 2.0)]
    open_run = np.r_[np.full(28, 2.0), np.zeros(5)]
    quiet = np.r_[np.full(28, 0.1), np.zeros(5), np.full(5, 0.1)]
    flags = flag_stockouts(np.vstack([busy, quiet]))
    assert flags[0, 28:33].all() and flags[0].sum() == 5
    assert not flags[1].any()
    assert not flag_stockouts(open_run[None, :]).any()


def test_calendar_marks_black_friday_and_brazilian_holidays():
    assert black_friday(2024) == pd.Timestamp("2024-11-29")
    dates = pd.date_range("2024-11-24", "2024-12-26")
    cal = calendar_features(dates).set_index(dates)
    assert (
        cal.loc["2024-11-25", "bf_week"] == 1 and cal.loc["2024-11-25", "bf_peak"] == 0
    )
    assert (
        cal.loc["2024-11-29", "bf_peak"] == 1 and cal.loc["2024-12-03", "bf_week"] == 0
    )
    assert cal.loc["2024-12-25", "br_holiday"] == 1
    assert cal.loc["2024-12-10", "xmas_runup"] == 1


def test_resolve_models_expands_groups_and_patterns():
    assert resolve_models("baselines")[:3] == [
        "seasonal_naive_7",
        "weekday_mean_4w",
        "moving_average_28",
    ]
    powers = resolve_models("lgbm_direct_p1.*")
    assert powers and all(name in LGBM_VARIANTS for name in powers)
    with pytest.raises(ValueError):
        resolve_models("no_such_model")


def test_run_fold_smoke_with_challengers(orders, monkeypatch):
    """End-to-end fold on a small sample with fast LightGBM variants."""
    fast = {
        name: replace(LGBM_VARIANTS[name], **FAST)
        for name in ("lgbm_direct", "lgbm_recursive")
    }
    monkeypatch.setattr("backtesting.run_backtest.LGBM_VARIANTS", fast)
    actuals = build_actuals_grid(orders)
    actuals_by_mlb = {m: f.set_index("date")["y"] for m, f in actuals.groupby("mlb")}
    truth = {m: s * 1.0 for m, s in actuals_by_mlb.items()}

    rows = run_fold(
        orders,
        actuals_by_mlb,
        CUTOFF,
        HORIZON,
        max_series=5,
        models=["weekday_mean_4w", *fast],
        truth_by_mlb=truth,
    )

    assert set(rows["model"]) == {"weekday_mean_4w", *fast}
    per_model = rows.groupby("model")["mlb"].nunique()
    assert per_model.nunique() == 1  # Challengers cover every eligible series
    assert rows["forecast"].notna().all() and (rows["forecast"] >= 0).all()
    assert rows["h"].between(1, HORIZON).all()
    np.testing.assert_allclose(rows["true_demand"], rows["actual"])
