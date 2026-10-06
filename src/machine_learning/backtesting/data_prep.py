"""
Load order exports and prepare them for backtesting.

Training data goes through the production cleaning and feature code unchanged
(``process_sales_data`` and ``create_time_series_features``). Scoring uses a separate
actuals grid: daily units per series, zero-filled from each series' first sale to the
last date in the export.
"""

import logging
from pathlib import Path
from typing import Union

import pandas as pd

from config import AppConfig
from data_management.clean_sql_data import process_sales_data
from data_management.feature_creation import create_time_series_features

logger = logging.getLogger(__name__)


def load_orders(source: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
    """
    Read an order export and normalise it to the production view's columns.

    Older exports such as ``raw_sql.csv`` are SKU-level and have no ``mlb`` column; for
    those the SKU is used as the series identifier. A ``date`` column (naive, normalised
    to midnight) is added for fold splitting.
    """
    orders = (
        source.copy()
        if isinstance(source, pd.DataFrame)
        else pd.read_csv(source, low_memory=False)
    )
    if "mlb" not in orders.columns:
        logger.info("No 'mlb' column found; using SKU as the series identifier")
        orders["mlb"] = orders["order_items_item_seller_sku"]
    orders["mlb"] = orders["mlb"].astype(str)
    # CSV exports hold mixed-format timestamp strings; parse them to UTC datetimes, as
    # the database driver returns them, so the production cleaner handles them as usual
    orders["date_created"] = pd.to_datetime(
        orders["date_created"], utc=True, format="mixed"
    )
    orders["date"] = orders["date_created"].dt.tz_localize(None).dt.normalize()
    return orders


def build_training_features(orders: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """
    Clean and feature-engineer orders up to and including the cutoff date.

    Cleaning runs on the truncated orders so that no information from after the cutoff
    (for example, how long a zero-sales spell eventually lasts) reaches training.
    """
    truncated = orders[orders["date"] <= cutoff].drop(columns="date")
    cleaned = process_sales_data(truncated)
    features = create_time_series_features(cleaned)
    if features.index.max() > cutoff:
        raise ValueError("Training features contain dates after the cutoff")
    return features


def eligible_series(
    features: pd.DataFrame, cutoff: pd.Timestamp, horizon: int
) -> list[str]:
    """Series the production pipeline would forecast at this cutoff."""
    activity_cutoff = cutoff - pd.Timedelta(days=AppConfig.ACTIVE_MLB_DAYS_THRESHOLD)
    stats = features.groupby("mlb").apply(
        lambda frame: pd.Series({"last": frame.index.max(), "rows": len(frame)}),
        include_groups=False,
    )
    keep = (stats["last"] >= activity_cutoff) & (stats["rows"] >= horizon + 15)
    return stats.index[keep].tolist()


def build_actuals_grid(orders: pd.DataFrame) -> pd.DataFrame:
    """
    Daily units per series in long format (``mlb``, ``date``, ``y``).

    Each series runs from its first sale to the last date in the export, with days
    without sales set to zero.
    """
    daily = orders.groupby(["mlb", "date"])["order_items_quantity"].sum()
    last_date = orders["date"].max()

    frames = []
    for mlb, series in daily.groupby(level="mlb"):
        series = series.droplevel("mlb")
        full_range = pd.date_range(series.index.min(), last_date, freq="D")
        frames.append(
            pd.DataFrame(
                {
                    "mlb": mlb,
                    "date": full_range,
                    "y": series.reindex(full_range, fill_value=0).to_numpy(),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)
