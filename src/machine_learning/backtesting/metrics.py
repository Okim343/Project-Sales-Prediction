"""
Accuracy metrics for backtest results.

All functions take a long-format frame with one row per (model, cutoff, mlb, date) and
columns ``forecast``, ``actual``, and ``h`` (1-based horizon step). MASE also needs a
per-(cutoff, mlb) ``scale``: the in-sample mean absolute error of a 7-day seasonal naive
forecast on the training history.
"""

import numpy as np
import pandas as pd

HORIZON_BUCKETS = [(1, 7, "days 1-7"), (8, 30, "days 8-30"), (31, 10_000, "days 31+")]


def wape(forecast: pd.Series, actual: pd.Series) -> float:
    """Weighted absolute percentage error: sum |forecast - actual| / sum actual."""
    total = actual.sum()
    return float(np.abs(forecast - actual).sum() / total) if total > 0 else float("nan")


def bias(forecast: pd.Series, actual: pd.Series) -> float:
    """Relative bias: sum (forecast - actual) / sum actual. Positive means over-forecast."""
    total = actual.sum()
    return float((forecast - actual).sum() / total) if total > 0 else float("nan")


def mase(results: pd.DataFrame) -> float:
    """
    Mean absolute scaled error, averaged over (cutoff, series) pairs.

    Each pair's MAE is divided by its ``scale``. Pairs with a zero scale (flat training
    history) are skipped.
    """
    per_series = results.groupby(["cutoff", "mlb"]).agg(
        mae=("abs_error", "mean"), scale=("scale", "first")
    )
    per_series = per_series[per_series["scale"] > 0]
    return (
        float((per_series["mae"] / per_series["scale"]).mean())
        if len(per_series)
        else float("nan")
    )


def seasonal_naive_scale(history: pd.Series, season: int = 7) -> float:
    """In-sample MAE of a seasonal naive forecast, used as the MASE denominator."""
    values = history.to_numpy(dtype=float)
    if len(values) <= season:
        return float("nan")
    return float(np.abs(values[season:] - values[:-season]).mean())


def summarise(results: pd.DataFrame) -> pd.DataFrame:
    """
    Scores per model overall, by horizon bucket, and by volume tier.

    ``results`` must also carry a ``tier`` column (for example "top 20%" / "rest").

    Returns:
        DataFrame with columns model, group_type, group, wape, mase, bias, series_x_folds
        (number of scored series-cutoff pairs).
    """
    results = results.assign(abs_error=(results["forecast"] - results["actual"]).abs())
    rows = []

    def add(model, group_type, group, frame):
        rows.append(
            {
                "model": model,
                "group_type": group_type,
                "group": group,
                "wape": wape(frame["forecast"], frame["actual"]),
                "mase": mase(frame),
                "bias": bias(frame["forecast"], frame["actual"]),
                "series_x_folds": frame[["cutoff", "mlb"]].drop_duplicates().shape[0],
            }
        )

    for model, frame in results.groupby("model"):
        add(model, "overall", "all", frame)
        for low, high, label in HORIZON_BUCKETS:
            add(model, "horizon", label, frame[frame["h"].between(low, high)])
        for tier, tier_frame in frame.groupby("tier"):
            add(model, "volume", tier, tier_frame)

    return pd.DataFrame(rows)
