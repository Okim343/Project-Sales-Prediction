"""Tests for baseline forecasts."""

import numpy as np
import pandas as pd

from backtesting.baselines import moving_average_28, seasonal_naive_7, weekday_mean_4w

CUTOFF = pd.Timestamp("2025-03-02")  # A Sunday


def _history(values):
    return pd.Series(
        np.asarray(values, dtype=float),
        index=pd.date_range(end=CUTOFF, periods=len(values), freq="D"),
    )


def test_forecasts_start_the_day_after_cutoff():
    history = _history(range(28))
    for baseline in (seasonal_naive_7, weekday_mean_4w, moving_average_28):
        forecast = baseline(history, CUTOFF, 10)
        assert forecast.index[0] == CUTOFF + pd.Timedelta(days=1)
        assert len(forecast) == 10


def test_seasonal_naive_repeats_last_week():
    history = _history([0] * 21 + [1, 2, 3, 4, 5, 6, 7])
    forecast = seasonal_naive_7(history, CUTOFF, 10)
    assert forecast.tolist() == [1, 2, 3, 4, 5, 6, 7, 1, 2, 3]


def test_weekday_mean_uses_same_weekday():
    # Mondays sell 10, other days 0, over the last four weeks
    history = _history([0] * 28)
    history[history.index.dayofweek == 0] = 10
    forecast = weekday_mean_4w(history, CUTOFF, 7)
    assert forecast[forecast.index.dayofweek == 0].iloc[0] == 10
    assert forecast[forecast.index.dayofweek != 0].sum() == 0


def test_moving_average_is_flat_28_day_mean():
    history = _history([100] * 10 + [2] * 28)
    forecast = moving_average_28(history, CUTOFF, 5)
    assert (forecast == 2).all()
