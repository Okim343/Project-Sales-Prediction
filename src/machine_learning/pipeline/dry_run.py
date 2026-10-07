"""Offline routing exercise that writes local parquet tables and JSON metadata."""

import json
import logging
import resource
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import joblib

from config import AppConfig, BLD
from database_utils import db_manager
from data_management.clean_sql_data import process_sales_data
from data_management.feature_creation import create_time_series_features
from estimation.lgbm_forecast import normalize_orders
from estimation.eligibility import eligible_series
from estimation.model_forecast import forecast_future_sales_direct
from pipeline.model_routing import run_model_routing

logger = logging.getLogger(__name__)


def _peak_memory_mb() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (1024 * 1024) if sys.platform == "darwin" else value / 1024


def run_dry_run(
    csv_path: Path,
    mode: str,
    since_date: str | None = None,
    max_legacy_series: int = 1,
) -> dict:
    """Run model routing from CSV with no SQL reads, writes, or metadata calls.

    A daily CSV run trains legacy XGBoost on the available history as a bootstrap;
    actual production daily continuation remains in ``run_daily_mode``.
    """
    started = time.monotonic()
    raw = pd.read_csv(csv_path, low_memory=False)
    normalized = normalize_orders(raw)
    as_of = normalized["date"].max() + pd.Timedelta(days=1)
    lgbm_window = normalized[
        normalized["date"]
        >= as_of - pd.DateOffset(months=AppConfig.LGBM_HISTORY_MONTHS)
    ].drop(columns="date")
    lgbm_history_orders = len(lgbm_window)
    history_holder = {"orders": lgbm_window}
    del lgbm_window, normalized
    out_dir = (
        BLD
        / "dry_run"
        / f"{datetime.now():%Y%m%d_%H%M%S_%f}_{mode}_{AppConfig.PRIMARY_MODEL}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = []
    tables = []
    legacy_cache_reused = False

    def save_table(forecasts, table):
        frame = db_manager.forecasts_frame(forecasts)
        path = out_dir / f"{table.split('.')[-1]}.parquet"
        frame.to_parquet(path, index=False)
        tables.append(str(path))

    def log_run(**record):
        metadata.append(record)
        (out_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, default=str)
        )

    def legacy_run():
        nonlocal legacy_cache_reused
        data = raw
        if mode in {"since_date", "daily"} and since_date:
            logger.info(
                "Dry-run legacy bootstrap uses full CSV history before %s", since_date
            )
        features = create_time_series_features(process_sales_data(data))
        if max_legacy_series:
            eligible = eligible_series(
                features, as_of - pd.Timedelta(days=1), AppConfig.FORECAST_DAYS_LONG
            )
            top = (
                features[features.mlb.isin(eligible)]
                .groupby("mlb")["quant"]
                .sum()
                .nlargest(max_legacy_series)
                .index
            )
            features = features[features.mlb.isin(top)]
        stat = csv_path.stat()
        cache = (
            BLD
            / "dry_run"
            / (
                f"legacy_{stat.st_size}_{stat.st_mtime_ns}_{max_legacy_series}_{as_of:%Y%m%d}.joblib"
            )
        )
        if cache.exists():
            forecasts, model_count = joblib.load(cache)
            legacy_cache_reused = True
        else:
            forecasts, models = forecast_future_sales_direct(
                features, AppConfig.FORECAST_DAYS_LONG, as_of_date=as_of
            )
            model_count = len(models)
            joblib.dump((forecasts, model_count), cache)
        return forecasts, {
            "run_type": {"since_date": "incremental_filtered"}.get(mode, mode),
            "status": "success",
            "records_processed": len(data),
            "models_updated": model_count,
        }

    result = run_model_routing(
        mode,
        legacy_run,
        orders_loader=lambda: history_holder.pop("orders"),
        save_table=save_table,
        read_previous=lambda table: None,
        log_run=log_run,
        as_of_date=as_of,
    )
    report = {
        **result,
        "mode": mode,
        "primary_model": AppConfig.PRIMARY_MODEL,
        "orders": len(raw),
        "lgbm_history_orders": lgbm_history_orders,
        "legacy_series_limit": max_legacy_series or "all",
        "legacy_cache_reused": legacy_cache_reused,
        "tables": tables,
        "runtime_seconds": round(time.monotonic() - started, 2),
        "peak_memory_mb": round(_peak_memory_mb(), 1),
        "output_dir": str(out_dir),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    logger.info("Dry-run report: %s", report)
    print(json.dumps(report, indent=2))
    if result["status"] == "failed":
        raise RuntimeError("Both dry-run forecast paths failed")
    return report
