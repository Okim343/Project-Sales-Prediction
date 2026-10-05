"""Rolling-origin fold cutoffs for backtesting."""

import logging
from typing import List

import pandas as pd

logger = logging.getLogger(__name__)


def make_cutoffs(
    min_date: pd.Timestamp,
    max_date: pd.Timestamp,
    horizon: int = 90,
    n_folds: int = 4,
    step_days: int = 30,
    min_train_days: int = 180,
) -> List[pd.Timestamp]:
    """
    Cutoff dates for rolling-origin evaluation, oldest first.

    The last cutoff leaves exactly ``horizon`` days of actuals after it; earlier cutoffs
    step back by ``step_days``. Cutoffs with fewer than ``min_train_days`` of history are
    dropped.

    Args:
        min_date: First date in the data.
        max_date: Last date in the data.
        horizon: Forecast length in days.
        n_folds: Number of cutoffs requested.
        step_days: Spacing between cutoffs.
        min_train_days: Minimum history required before a cutoff.

    Returns:
        List of cutoff timestamps. Training uses dates <= cutoff; scoring uses
        cutoff + 1 .. cutoff + horizon.
    """
    min_date = pd.Timestamp(min_date).normalize()
    last_cutoff = pd.Timestamp(max_date).normalize() - pd.Timedelta(days=horizon)
    cutoffs = [
        last_cutoff - pd.Timedelta(days=step_days * k) for k in reversed(range(n_folds))
    ]
    valid = [c for c in cutoffs if (c - min_date).days >= min_train_days]
    if len(valid) < len(cutoffs):
        logger.warning(
            f"Dropped {len(cutoffs) - len(valid)} cutoff(s) with less than "
            f"{min_train_days} days of history"
        )
    if not valid:
        raise ValueError("No cutoff has enough training history")
    return valid


def horizon_dates(cutoff: pd.Timestamp, horizon: int) -> pd.DatetimeIndex:
    """The dates scored for a cutoff: the ``horizon`` days after it."""
    return pd.date_range(cutoff + pd.Timedelta(days=1), periods=horizon, freq="D")
