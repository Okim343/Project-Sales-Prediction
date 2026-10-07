"""Shared production and backtest listing eligibility rule."""

import pandas as pd

from config import AppConfig


def eligible_series(
    features: pd.DataFrame, cutoff: pd.Timestamp, horizon: int
) -> list[str]:
    """Return recently active listings with at least ``horizon + 15`` clean days."""
    activity_cutoff = cutoff - pd.Timedelta(days=AppConfig.ACTIVE_MLB_DAYS_THRESHOLD)
    stats = features.groupby("mlb").apply(
        lambda frame: pd.Series({"last": frame.index.max(), "rows": len(frame)}),
        include_groups=False,
    )
    keep = (stats["last"] >= activity_cutoff) & (stats["rows"] >= horizon + 15)
    return stats.index[keep].tolist()
