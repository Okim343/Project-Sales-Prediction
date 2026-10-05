"""
Simple benchmark forecasts.

Each baseline takes a series' zero-filled daily history (indexed by date, ending on the
cutoff) and returns a forecast for the ``horizon`` days after the cutoff.
"""

from typing import Callable, Dict

import numpy as np
import pandas as pd

from backtesting.folds import horizon_dates


def seasonal_naive_7(
    history: pd.Series, cutoff: pd.Timestamp, horizon: int
) -> pd.Series:
    """Repeat the last 7 days."""
    last_week = history.iloc[-7:].to_numpy()
    last_week = np.pad(last_week, (7 - len(last_week), 0))
    values = np.resize(last_week, horizon)
    return pd.Series(values.astype(float), index=horizon_dates(cutoff, horizon))


def weekday_mean_4w(
    history: pd.Series, cutoff: pd.Timestamp, horizon: int
) -> pd.Series:
    """For each weekday, the mean of that weekday over the last 4 weeks."""
    recent = history.iloc[-28:]
    profile = recent.groupby(recent.index.dayofweek).mean()
    dates = horizon_dates(cutoff, horizon)
    values = profile.reindex(dates.dayofweek).fillna(recent.mean()).to_numpy()
    return pd.Series(values.astype(float), index=dates)


def moving_average_28(
    history: pd.Series, cutoff: pd.Timestamp, horizon: int
) -> pd.Series:
    """Flat forecast equal to the mean of the last 28 days."""
    return pd.Series(
        float(history.iloc[-28:].mean()), index=horizon_dates(cutoff, horizon)
    )


BASELINES: Dict[str, Callable[[pd.Series, pd.Timestamp, int], pd.Series]] = {
    "seasonal_naive_7": seasonal_naive_7,
    "weekday_mean_4w": weekday_mean_4w,
    "moving_average_28": moving_average_28,
}
