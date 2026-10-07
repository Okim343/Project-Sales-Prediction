"""Tests for protection-window forecasts and replenishment scoring."""

from argparse import Namespace

import pandas as pd
import pytest

from backtesting.data_prep import build_actuals_grid, load_orders
from backtesting.replenishment import (
    _inner_window_rows,
    apply_spread,
    direct_window_quantiles,
    fit_dispersion,
    inner_origins,
    nb_quantile,
    score_replenishment,
    window_totals,
)
from backtesting.run_backtest import run_source
from backtesting.synthetic_data import generate_synthetic_orders

CUTOFF = pd.Timestamp("2025-01-31")


@pytest.fixture(scope="module")
def sample():
    raw, _ = generate_synthetic_orders(
        n_skus=20, start="2024-01-01", end="2025-03-31", seed=5
    )
    orders = load_orders(raw)
    actuals = build_actuals_grid(orders)
    by_mlb = {
        mlb: frame.set_index("date")["y"] for mlb, frame in actuals.groupby("mlb")
    }
    return orders, by_mlb


def test_window_sums():
    cutoff = pd.Timestamp("2025-01-01")
    daily = pd.DataFrame(
        {
            "model": ["x"] * 3,
            "cutoff": [cutoff] * 3,
            "mlb": ["A"] * 3,
            "h": [1, 2, 3],
            "forecast": [1.5, 2.5, 4],
            "actual": [3, 0, 2],
            "tier": ["rest"] * 3,
            "in_common_set": [True] * 3,
        }
    )
    history = pd.Series(
        [1] * 28 + [3, 0, 2],
        index=pd.date_range(cutoff - pd.Timedelta(days=27), periods=31),
    )
    totals = window_totals(daily, [2, 3], {"A": history})
    assert totals.mu_W.tolist() == [4, 8]
    assert totals.actual.tolist() == [3, 5]


def test_inner_origins_and_inner_eligibility(sample, monkeypatch):
    orders, actuals = sample
    origins = inner_origins(CUTOFF, 14)
    assert all(origin + pd.Timedelta(days=14) <= CUTOFF for origin in origins)
    selected = []
    from backtesting import replenishment

    original = replenishment.eligible_series

    def record(features, origin, horizon):
        selected.append(origin)
        return original(features, origin, horizon)

    monkeypatch.setattr(replenishment, "eligible_series", record)
    rows = _inner_window_rows(orders, actuals, CUTOFF, [7, 14], ["weekday_mean_4w"], 90)
    assert set(selected) == set(origins)
    assert (rows.origin + pd.to_timedelta(rows.L, unit="D") <= CUTOFF).all()


def test_nb_quantile_shape_and_poisson_limit():
    from scipy.stats import poisson

    assert nb_quantile(0, 4, 0.95) == 0
    assert nb_quantile(3, 1e-12, 0.9) == poisson.ppf(0.9, 3)
    values = [nb_quantile(3, 0.7, alpha) for alpha in (0.5, 0.8, 0.9, 0.95)]
    assert values == sorted(values) and min(values) >= 0


def test_spread_is_leakage_free(sample):
    orders, actuals = sample
    models = ["weekday_mean_4w"]
    original = _inner_window_rows(orders, actuals, CUTOFF, [7], models, 90)
    changed = orders.copy()
    after = changed.date > CUTOFF
    changed.loc[after, "order_items_quantity"] *= 100
    changed.loc[after, "order_items_unit_price"] *= 10
    altered = _inner_window_rows(changed, actuals, CUTOFF, [7], models, 90)
    pd.testing.assert_frame_equal(original, altered)
    base = fit_dispersion(original, CUTOFF, minimum=3)
    other = fit_dispersion(altered, CUTOFF, minimum=3)
    pd.testing.assert_frame_equal(base, other)
    outer = pd.DataFrame(
        [
            {
                "model": models[0],
                "cutoff": CUTOFF,
                "mlb": "A",
                "L": 7,
                "tier": "rest",
                "mu_W": 3.0,
                "actual": 2.0,
                "in_common_set": True,
                "trailing_28": 0.5,
            }
        ]
    )
    pd.testing.assert_frame_equal(
        apply_spread(outer, base, ["nb_tier"]),
        apply_spread(outer, other, ["nb_tier"]),
    )


def test_direct_quantile_is_leakage_free(sample):
    orders, _ = sample
    mlbs = orders[orders.date <= CUTOFF].mlb.unique()[:3].tolist()
    base = direct_window_quantiles(
        orders,
        CUTOFF,
        [7],
        [0.5, 0.9],
        mlbs,
        fast={"n_estimators": 10, "min_child_samples": 5},
    )
    changed = orders.copy()
    after = changed.date > CUTOFF
    changed.loc[after, "order_items_quantity"] *= 100
    changed.loc[after, "order_items_unit_price"] *= 10
    other = direct_window_quantiles(
        changed,
        CUTOFF,
        [7],
        [0.5, 0.9],
        mlbs,
        fast={"n_estimators": 10, "min_child_samples": 5},
    )
    pd.testing.assert_frame_equal(base, other)
    assert (base["q_0.5"] >= 0).all()
    assert (base["q_0.5"] <= base["q_0.9"]).all()


def test_hand_computed_metrics():
    frame = pd.DataFrame(
        {
            "model": ["perfect", "perfect", "low", "low"],
            "method": ["nb_tier"] * 4,
            "L": [7] * 4,
            "cutoff": [CUTOFF] * 4,
            "tier": ["rest"] * 4,
            "mu_W": [2, 4, 1, 1],
            "actual": [2, 4, 2, 4],
            "trailing_28": [1] * 4,
            "q_0.9": [2, 4, 1, 1],
        }
    )
    score = (
        score_replenishment(frame, [0.9]).query("scope == 'overall'").set_index("model")
    )
    assert score.loc["perfect", "coverage"] == 1
    assert score.loc["perfect", "scaled_pinball"] == 0
    assert score.loc["perfect", "fill_rate"] == 1
    assert score.loc["low", "coverage"] == 0
    assert score.loc["low", "fill_rate"] == pytest.approx(1 / 3)
    assert score.loc["low", "scaled_pinball"] == pytest.approx(0.6)


def test_run_source_smoke_creates_window_files(sample, monkeypatch, tmp_path):
    orders, _ = sample
    from backtesting import run_backtest

    monkeypatch.setattr(run_backtest, "PROJECT_ROOT", tmp_path)
    args = Namespace(
        models="weekday_mean_4w",
        skip_current=False,
        horizon=14,
        folds=1,
        step_days=30,
        max_series=5,
        replenishment=True,
        windows="7",
        service_levels="0.5,0.9",
        quantile_methods="nb_tier,ratio_tier",
        replenishment_include_current=False,
    )
    folder = run_source("synthetic", orders, args)
    for filename in (
        "window_forecasts.parquet",
        "dispersion.csv",
        "replenishment_scores.csv",
    ):
        assert (folder / filename).exists()
    windows = pd.read_parquet(folder / "window_forecasts.parquet")
    assert {"model", "method", "cutoff", "mlb", "L", "mu_W", "q_0.9", "actual"} <= set(
        windows
    )
    scores = pd.read_csv(folder / "replenishment_scores.csv")
    assert {"coverage", "scaled_pinball", "fill_rate", "excess_units"} <= set(scores)
