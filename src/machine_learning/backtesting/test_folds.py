"""Tests for fold cutoffs and leakage protection."""

import pandas as pd
import pytest

from backtesting.data_prep import build_training_features, load_orders
from backtesting.folds import horizon_dates, make_cutoffs
from backtesting.synthetic_data import generate_synthetic_orders


def test_cutoffs_match_historical_export_dates():
    cutoffs = make_cutoffs(pd.Timestamp("2023-12-24"), pd.Timestamp("2025-03-16"))
    assert [c.date().isoformat() for c in cutoffs] == [
        "2024-09-17",
        "2024-10-17",
        "2024-11-16",
        "2024-12-16",
    ]


def test_last_cutoff_leaves_exactly_one_horizon():
    max_date = pd.Timestamp("2025-03-16")
    cutoff = make_cutoffs(pd.Timestamp("2023-01-01"), max_date, horizon=30)[-1]
    assert horizon_dates(cutoff, 30)[-1] == max_date


def test_short_history_cutoffs_are_dropped():
    cutoffs = make_cutoffs(
        pd.Timestamp("2025-01-01"),
        pd.Timestamp("2025-12-31"),
        n_folds=6,
        min_train_days=200,
    )
    assert all((c - pd.Timestamp("2025-01-01")).days >= 200 for c in cutoffs)
    assert len(cutoffs) < 6


def test_no_cutoff_with_enough_history_raises():
    with pytest.raises(ValueError):
        make_cutoffs(pd.Timestamp("2025-01-01"), pd.Timestamp("2025-04-30"))


def test_training_features_stop_at_cutoff():
    orders, _ = generate_synthetic_orders(
        n_skus=15, start="2024-06-01", end="2025-03-31", seed=1
    )
    cutoff = pd.Timestamp("2025-01-15")
    features = build_training_features(load_orders(orders), cutoff)
    assert features.index.max() <= cutoff
    assert {"lag_1", "rolling_mean_3", "day_of_week", "day_of_month"} <= set(
        features.columns
    )
