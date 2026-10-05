"""Tests for backtest metrics."""

import numpy as np
import pandas as pd
import pytest

from backtesting.metrics import bias, mase, seasonal_naive_scale, summarise, wape


def test_wape_and_bias_by_hand():
    forecast = pd.Series([2.0, 0.0, 5.0])
    actual = pd.Series([1.0, 2.0, 5.0])
    # |e| = 1, 2, 0 -> 3 / 8 ; e = 1, -2, 0 -> -1 / 8
    assert wape(forecast, actual) == pytest.approx(3 / 8)
    assert bias(forecast, actual) == pytest.approx(-1 / 8)


def test_wape_with_no_sales_is_nan():
    assert np.isnan(wape(pd.Series([1.0]), pd.Series([0.0])))


def test_seasonal_naive_scale():
    history = pd.Series([1, 2, 3, 4, 5, 6, 7, 3, 2, 3, 4, 5, 6, 7], dtype=float)
    # Differences at lag 7: 2, 0, 0, 0, 0, 0, 0 -> mean 2/7
    assert seasonal_naive_scale(history) == pytest.approx(2 / 7)


def test_mase_averages_over_series_and_skips_zero_scale():
    results = pd.DataFrame(
        {
            "cutoff": ["c1"] * 4,
            "mlb": ["a", "a", "b", "b"],
            "abs_error": [2.0, 4.0, 1.0, 1.0],
            "scale": [2.0, 2.0, 0.0, 0.0],
        }
    )
    # Series a: MAE 3 / scale 2 = 1.5 ; series b is skipped
    assert mase(results) == pytest.approx(1.5)


def test_summarise_produces_all_groups():
    results = pd.DataFrame(
        {
            "model": "m",
            "cutoff": "c1",
            "mlb": ["a"] * 40,
            "h": range(1, 41),
            "forecast": 1.0,
            "actual": 2.0,
            "scale": 1.0,
            "tier": "rest",
        }
    )
    scores = summarise(results)
    assert set(scores["group_type"]) == {"overall", "horizon", "volume"}
    overall = scores[scores["group_type"] == "overall"].iloc[0]
    assert overall["wape"] == pytest.approx(0.5)
    assert overall["bias"] == pytest.approx(-0.5)
    assert overall["mase"] == pytest.approx(1.0)
