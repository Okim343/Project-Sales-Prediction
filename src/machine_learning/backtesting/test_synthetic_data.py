"""Tests for the synthetic order generator."""

import numpy as np
import pandas as pd
import pytest

from backtesting.data_prep import load_orders
from backtesting.synthetic_data import (
    DAY_OF_WEEK_PROFILE,
    VIEW_COLUMNS,
    calibration_report,
    generate_synthetic_orders,
)


@pytest.fixture(scope="module")
def synthetic():
    return generate_synthetic_orders(
        n_skus=120, start="2024-01-01", end="2025-06-30", seed=7
    )


def test_orders_have_view_schema(synthetic):
    orders, _ = synthetic
    assert list(orders.columns) == VIEW_COLUMNS
    assert orders["order_id"].is_unique
    assert (orders["order_items_quantity"] >= 1).all()
    assert orders["mlb"].str.startswith("MLB").all()


def test_same_seed_gives_identical_output():
    first, _ = generate_synthetic_orders(
        n_skus=20, start="2025-01-01", end="2025-03-31", seed=3
    )
    second, _ = generate_synthetic_orders(
        n_skus=20, start="2025-01-01", end="2025-03-31", seed=3
    )
    pd.testing.assert_frame_equal(first, second)


def test_orders_sum_to_observed_sales(synthetic):
    orders, truth = synthetic
    assert orders["order_items_quantity"].sum() == truth["daily"]["observed"].sum()


def test_stockouts_censor_observed_sales(synthetic):
    _, truth = synthetic
    daily = truth["daily"]
    stockout_days = daily[daily["stockout"]]
    assert len(stockout_days) > 0
    assert (stockout_days["observed"] == 0).all()
    assert stockout_days["true_demand"].sum() > 0


def test_weekday_profile_is_recovered(synthetic):
    orders, _ = synthetic
    report = calibration_report(orders)
    days = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
    recovered = np.array([report[f"weekday_ratio_{d}"] for d in days])
    expected = DAY_OF_WEEK_PROFILE / DAY_OF_WEEK_PROFILE.mean()
    assert np.abs(recovered - expected).max() < 0.12


def test_sales_are_intermittent(synthetic):
    orders, _ = synthetic
    report = calibration_report(orders)
    assert 0.5 < report["zero_share_all"] < 0.95


def test_some_skus_have_two_listings(synthetic):
    _, truth = synthetic
    listings_per_sku = truth["params"].groupby("sku")["mlb"].nunique()
    assert (listings_per_sku == 2).any()


def test_load_orders_parses_synthetic_export(synthetic):
    orders, _ = synthetic
    loaded = load_orders(orders)
    assert loaded["date"].dt.tz is None
    assert (loaded["date"] == loaded["date"].dt.normalize()).all()
