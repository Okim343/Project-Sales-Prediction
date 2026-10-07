"""Run both forecasters and publish the selected forecast with fallback."""

import logging
import gc
import time
from urllib.parse import quote, quote_plus

import numpy as np
import pandas as pd

from config import AppConfig, DatabaseConfig
from database_utils import db_manager
from data_management.import_SQL import import_data_last_n_months
from data_management.metadata_tracker import log_pipeline_run
from estimation.lgbm_forecast import forecast_lgbm_direct, normalize_orders

logger = logging.getLogger(__name__)


def _safe_error(exc: Exception) -> str:
    """Keep DB credentials out of metadata and fallback diagnostics."""
    detail = str(exc)
    password = DatabaseConfig.PASSWORD
    if password:
        for token in {password, quote(password, safe=""), quote_plus(password)}:
            detail = detail.replace(token, "[redacted]")
    return detail


def capture_legacy_run(run_callable, module_globals: dict) -> tuple[dict, dict]:
    """Collect a legacy mode's output without changing its training flow.

    Legacy modes used to publish directly into the consumer table. We intercept only
    that publication and metadata call while the mode executes. Its model training,
    validation, rollback, rounding, and pickle writes remain the original code.
    """
    forecasts = {}
    event = {}
    original_save = db_manager.save_forecasts_to_sql
    original_log = module_globals["log_pipeline_run"]

    def collect_save(value, table=None):
        nonlocal forecasts
        forecasts = value

    def collect_log(**kwargs):
        nonlocal event
        event = kwargs
        return True

    try:
        db_manager.save_forecasts_to_sql = collect_save
        module_globals["log_pipeline_run"] = collect_log
        run_callable()
    finally:
        db_manager.save_forecasts_to_sql = original_save
        module_globals["log_pipeline_run"] = original_log
    return forecasts, event


def import_lgbm_history() -> pd.DataFrame:
    """Read the same rolling history for LightGBM in every pipeline mode."""
    return import_data_last_n_months(
        user=DatabaseConfig.USER,
        password=DatabaseConfig.PASSWORD,
        host=DatabaseConfig.HOST,
        port=DatabaseConfig.PORT,
        dbname=DatabaseConfig.DBNAME,
        view=DatabaseConfig.VIEW,
        months=AppConfig.LGBM_HISTORY_MONTHS,
    )


def check_lgbm_forecasts(
    forecasts: dict,
    bundle: dict,
    orders: pd.DataFrame,
    previous: pd.DataFrame | None = None,
    as_of_date=None,
) -> dict[str, float | bool]:
    """Evaluate completeness, coverage, level, and prior-run drift."""
    as_of = pd.Timestamp(as_of_date or pd.Timestamp.now()).normalize()
    cutoff = as_of - pd.Timedelta(days=1)
    expected = pd.date_range(
        as_of + pd.Timedelta(days=1), periods=AppConfig.FORECAST_DAYS_LONG
    )
    eligible = set(bundle["eligible_mlbs"])
    finite = all(
        np.isfinite(frame["prediction"].to_numpy(dtype=float)).all()
        and (frame["prediction"] >= 0).all()
        for frame, _ in forecasts.values()
    )
    dates = all(frame.index.equals(expected) for frame, _ in forecasts.values())
    coverage = (
        len(eligible.intersection(forecasts)) / len(eligible) if eligible else 0.0
    )
    recent = normalize_orders(orders)
    recent = recent[
        recent["mlb"].isin(eligible)
        & recent["date"].between(cutoff - pd.Timedelta(days=27), cutoff)
    ]
    actual = float(recent["order_items_quantity"].sum())
    predicted = sum(
        float(frame["prediction"].iloc[:28].sum())
        for mlb, (frame, _) in forecasts.items()
        if mlb in eligible
    )
    level = predicted / actual if actual > 0 else float("nan")
    drift = float("nan")
    drift_ok = True
    if previous is not None and not previous.empty:
        prior = previous.copy()
        prior["date"] = pd.to_datetime(prior["date"]).dt.normalize()
        overlap = prior[
            prior["date"].isin(expected[:28]) & prior["mlb"].astype(str).isin(eligible)
        ]
        if not overlap.empty:
            current_by_date = pd.Series(
                {
                    day: sum(
                        float(frame.loc[day, "prediction"])
                        for mlb, (frame, _) in forecasts.items()
                        if mlb in eligible
                    )
                    for day in expected[:28]
                }
            )
            overlap_dates = pd.DatetimeIndex(overlap["date"].unique())
            prior_total = float(overlap["prediction"].sum())
            drift = (
                float(current_by_date.reindex(overlap_dates).sum() / prior_total)
                if prior_total > 0
                else float("nan")
            )
            drift_ok = AppConfig.LGBM_DRIFT_MIN <= drift <= AppConfig.LGBM_DRIFT_MAX
    checks = {
        "finite_nonnegative": bool(finite),
        "dates": bool(dates),
        "coverage": coverage,
        "coverage_ok": coverage >= AppConfig.LGBM_MIN_COVERAGE,
        "level": level,
        "level_ok": AppConfig.LGBM_LEVEL_MIN <= level <= AppConfig.LGBM_LEVEL_MAX,
        "drift": drift,
        "drift_ok": drift_ok,
    }
    for name, value in checks.items():
        logger.info("LightGBM sanity %s=%s", name, value)
    return checks


def run_model_routing(
    mode: str,
    legacy_run,
    *,
    orders_loader=import_lgbm_history,
    save_table=None,
    read_previous=None,
    log_run=log_pipeline_run,
    as_of_date=None,
) -> dict:
    """Publish side tables and an atomically selected main table for one run.

    ``legacy_run`` returns ``(forecasts, metadata_kwargs)`` and may raise. The
    legacy training/validation implementation stays in its existing mode function.
    """
    if AppConfig.PRIMARY_MODEL not in {"xgboost", "lgbm"}:
        raise ValueError("PRIMARY_MODEL must be xgboost or lgbm")
    save_table = save_table or db_manager.save_forecasts_atomic
    read_previous = read_previous or db_manager.read_forecasts
    started = time.monotonic()
    available = {}
    errors = {}
    lgbm_rows = 0
    try:
        orders = orders_loader()
        lgbm_rows = len(orders)
        previous = read_previous(DatabaseConfig.LGBM_FORECAST_TABLE)
        lgbm_forecasts, bundle = forecast_lgbm_direct(orders, as_of_date=as_of_date)
        checks = check_lgbm_forecasts(
            lgbm_forecasts, bundle, orders, previous, as_of_date
        )
        if not all(
            checks[k]
            for k in (
                "finite_nonnegative",
                "dates",
                "coverage_ok",
                "level_ok",
                "drift_ok",
            )
        ):
            raise ValueError(f"LightGBM sanity check failed: {checks}")
        save_table(lgbm_forecasts, DatabaseConfig.LGBM_FORECAST_TABLE)
        available["lgbm"] = lgbm_forecasts
        logger.info("LightGBM artifact: %s", bundle["path"])
    except Exception as exc:
        errors["lgbm"] = _safe_error(exc)
        logger.error("LightGBM path failed: %s", errors["lgbm"])
    finally:
        if "bundle" in locals():
            del bundle
        if "orders" in locals():
            del orders
        gc.collect()

    legacy_meta = {"run_type": mode, "records_processed": 0, "models_updated": 0}
    legacy_noop = False
    if AppConfig.RUN_LEGACY_MODEL:
        try:
            legacy_forecasts, event = legacy_run()
            if event:
                legacy_meta.update(event)
            if not legacy_forecasts:
                if (
                    legacy_meta.get("status") == "success"
                    and legacy_meta.get("models_updated", 0) == 0
                ):
                    legacy_noop = True
                    logger.info(
                        "Legacy run was a successful no-op; retaining its tables"
                    )
                else:
                    raise ValueError("Legacy model produced no forecasts")
            else:
                # Daily legacy forecasts contain only updated MLBs. This intentionally
                # preserves the old replace-table behavior and can shrink this table.
                save_table(legacy_forecasts, DatabaseConfig.LEGACY_FORECAST_TABLE)
                available["xgboost"] = legacy_forecasts
        except Exception as exc:
            errors["xgboost"] = _safe_error(exc)
            logger.error("Legacy XGBoost path failed: %s", errors["xgboost"])

    primary = AppConfig.PRIMARY_MODEL
    secondary = "lgbm" if primary == "xgboost" else "xgboost"
    chosen = (
        primary
        if primary in available
        else secondary
        if secondary in available
        else None
    )
    if primary == "xgboost" and legacy_noop:
        chosen = "unchanged"
    if chosen and chosen != "unchanged":
        try:
            save_table(available[chosen], DatabaseConfig.FORECAST_TABLE)
        except Exception as exc:
            errors["main"] = _safe_error(exc)
            chosen = None
    status = (
        "failed"
        if chosen is None
        else "partial"
        if errors or chosen not in {primary, "unchanged"}
        else "success"
    )
    if chosen is None:
        message = "main=unchanged; " + "; ".join(
            f"{name}: {reason}" for name, reason in errors.items()
        )
    elif chosen == "unchanged":
        message = "main=unchanged (legacy no-op)"
    elif chosen != primary:
        message = (
            f"main={chosen} (fallback: {errors.get(primary, 'primary unavailable')})"
        )
    else:
        message = f"main={chosen}" + (
            "; " + "; ".join(f"{name}: {reason}" for name, reason in errors.items())
            if errors
            else ""
        )
    elapsed = time.monotonic() - started
    log_run(
        run_type=f"{mode}_lgbm",
        status=status,
        records_processed=lgbm_rows,
        models_updated=len(available.get("lgbm", {})),
        error_message=message,
        run_duration_seconds=elapsed,
    )
    if AppConfig.RUN_LEGACY_MODEL:
        # This row retains the legacy run type and status for daily baseline lookup.
        log_run(
            run_type=legacy_meta["run_type"],
            status=legacy_meta.get("status", "success")
            if "xgboost" in available or legacy_noop
            else "failed",
            records_processed=legacy_meta.get("records_processed", 0),
            models_updated=legacy_meta.get("models_updated", 0),
            error_message=message,
            run_duration_seconds=elapsed,
        )
    logger.info("Model routing %s: %s", status, message)
    return {"status": status, "main": chosen, "message": message, "errors": errors}
