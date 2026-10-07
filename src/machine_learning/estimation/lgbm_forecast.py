"""Production adapter for the frozen global LightGBM direct model."""

import hashlib
import json
import logging
import warnings
from dataclasses import asdict
from pathlib import Path

import joblib
import lightgbm
import numpy as np
import pandas as pd

from config import AppConfig
from data_management.clean_sql_data import process_sales_data
from data_management.feature_creation import create_time_series_features
from estimation.eligibility import eligible_series
from estimation.lightgbm_model import DEFAULT_CONFIG, LightGBMFold, feature_columns

logger = logging.getLogger(__name__)


def normalize_orders(orders: pd.DataFrame) -> pd.DataFrame:
    """Keep each order's wall-clock day, as production cleaning does.

    The historical backtest's UTC conversion intentionally is not used here: the
    legacy production model assigns an order to the date returned by the DB driver.
    """
    data = orders.copy()
    if "mlb" not in data:
        raise ValueError("Production orders require an mlb column")
    data["mlb"] = data["mlb"].astype(str)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="In a future version of pandas, parsing datetimes with mixed time zones",
                category=FutureWarning,
            )
            parsed = pd.to_datetime(
                data["date_created"], errors="coerce", format="mixed"
            )
    except ValueError:
        parsed = data["date_created"]
    if isinstance(parsed.dtype, pd.DatetimeTZDtype):
        parsed = parsed.dt.tz_localize(None)
    elif not isinstance(parsed.dtype, np.dtype) or parsed.dtype.kind != "M":
        parsed = data["date_created"].map(
            lambda value: pd.Timestamp(value).tz_localize(None)
            if pd.notna(value)
            else pd.NaT
        )
    data["date_created"] = parsed
    data["date"] = parsed.dt.normalize()
    data = data.dropna(subset=["date", "mlb", "order_items_quantity"])
    return data


def save_lgbm_bundle(bundle: dict, directory: Path | None = None) -> Path:
    """Persist a fitted run and retain the newest fourteen cutoff artifacts."""
    directory = directory or AppConfig.LGBM_MODEL_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"model_{bundle['cutoff']}.joblib"
    joblib.dump(bundle, path)
    for old in sorted(directory.glob("model_*.joblib"), reverse=True)[14:]:
        old.unlink()
    return path


def forecast_lgbm_direct(
    orders: pd.DataFrame,
    as_of_date=None,
    horizon: int = AppConfig.FORECAST_DAYS_LONG,
) -> tuple[dict, dict]:
    """Train on complete days through yesterday; write tomorrow through day 90.

    Dates retain the database driver's wall-clock day, matching the legacy cleaner.
    The frozen final direct block also predicts day 91 from yesterday's origin;
    today's prediction is discarded because today is incomplete.
    """
    as_of = pd.Timestamp(as_of_date or pd.Timestamp.now()).normalize()
    cutoff = as_of - pd.Timedelta(days=1)
    normalized = normalize_orders(orders)
    history = normalized.loc[normalized["date"] <= cutoff].copy()
    if history.empty:
        raise ValueError("No complete-day orders available for LightGBM")
    if (cutoff - history["date"].min()).days < 365 + 91:
        logger.warning("LightGBM has less than the preferred training history")
    features = create_time_series_features(
        process_sales_data(history.drop(columns="date"))
    )
    eligible = eligible_series(features, cutoff, horizon)
    if not eligible:
        raise ValueError("No eligible MLBs for LightGBM")
    fold = LightGBMFold(history, cutoff, horizon + 1, retain_models=True)
    predicted = fold.forecast(DEFAULT_CONFIG, eligible)
    expected = pd.date_range(as_of + pd.Timedelta(days=1), periods=horizon)
    sku_column = "order_items_item_seller_sku"
    latest = history.sort_values("date_created").drop_duplicates("mlb", keep="last")
    skus = latest.set_index("mlb")[sku_column].to_dict() if sku_column in latest else {}
    forecasts = {}
    for mlb, series in predicted.items():
        future = series.iloc[1:]
        assert future.index.equals(expected), "LightGBM forecast date alignment failed"
        values = future.to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"Invalid LightGBM prediction for {mlb}")
        forecasts[mlb] = (
            pd.DataFrame({"prediction": values}, index=expected),
            skus.get(mlb),
        )
    config = asdict(DEFAULT_CONFIG)
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[
        :12
    ]
    bundle = {
        "models": fold.fitted_models,
        "config": config,
        "feature_columns": {
            str(block): feature_columns(DEFAULT_CONFIG, block)
            for block in DEFAULT_CONFIG.blocks
        },
        "cutoff": cutoff.date().isoformat(),
        "training_start": history["date"].min().date().isoformat(),
        "training_end": cutoff.date().isoformat(),
        "series_count": len(fold.panel.mlbs),
        "eligible_count": len(eligible),
        "order_rows": len(history),
        "training_rows": {
            str(block): count for block, count in fold.training_rows.items()
        },
        "lightgbm_version": lightgbm.__version__,
        "pandas_version": pd.__version__,
        "config_hash": digest,
        "eligible_mlbs": eligible,
    }
    bundle["path"] = str(save_lgbm_bundle(bundle))
    logger.info("Saved LightGBM bundle to %s", bundle["path"])
    return forecasts, bundle
