"""Routing and fallback tests use fixed-seed synthetic orders and no database."""

import pandas as pd
import pytest

from backtesting.synthetic_data import generate_synthetic_orders
from config import AppConfig, DatabaseConfig
from estimation.lgbm_forecast import normalize_orders
from pipeline import model_routing


@pytest.fixture(scope="module")
def sample():
    orders, _ = generate_synthetic_orders(
        n_skus=8, start="2024-01-01", end="2025-03-31", seed=19
    )
    normalized = normalize_orders(orders)
    as_of = pd.Timestamp("2025-04-01")
    cutoff = as_of - pd.Timedelta(days=1)
    recent = normalized[normalized.date.between(cutoff - pd.Timedelta(days=27), cutoff)]
    mlb = recent.groupby("mlb").order_items_quantity.sum().idxmax()
    level = recent.loc[recent.mlb == mlb, "order_items_quantity"].sum() / 28
    dates = pd.date_range(as_of + pd.Timedelta(days=1), periods=90)
    lgbm = {
        mlb: (pd.DataFrame({"prediction": [float(level)] * 90}, index=dates), "SKU")
    }
    legacy = {
        mlb: (
            pd.DataFrame({"prediction": [int(round(level))] * 90}, index=dates),
            "SKU",
        )
    }
    return orders, as_of, lgbm, legacy


def setup_route(monkeypatch, sample, primary="xgboost", legacy=True):
    orders, as_of, lgbm, xgb = sample
    monkeypatch.setattr(AppConfig, "PRIMARY_MODEL", primary)
    monkeypatch.setattr(AppConfig, "RUN_LEGACY_MODEL", legacy)
    monkeypatch.setattr(
        model_routing,
        "forecast_lgbm_direct",
        lambda value, as_of_date=None: (
            lgbm,
            {"eligible_mlbs": list(lgbm), "path": "/tmp/model.joblib"},
        ),
    )
    tables = {}
    records = []
    result = model_routing.run_model_routing(
        "daily",
        lambda: (xgb, {"run_type": "daily", "status": "success", "models_updated": 1}),
        orders_loader=lambda: orders,
        save_table=lambda forecast, table: tables.__setitem__(table, forecast),
        read_previous=lambda table: None,
        log_run=lambda **row: records.append(row),
        as_of_date=as_of,
    )
    return result, tables, records


@pytest.mark.parametrize("primary", ["xgboost", "lgbm"])
def test_primary_and_side_tables(monkeypatch, sample, primary):
    result, tables, records = setup_route(monkeypatch, sample, primary)
    assert result["status"] == "success"
    assert result["main"] == primary
    assert set(tables) == {
        DatabaseConfig.FORECAST_TABLE,
        DatabaseConfig.LGBM_FORECAST_TABLE,
        DatabaseConfig.LEGACY_FORECAST_TABLE,
    }
    expected = (
        DatabaseConfig.LEGACY_FORECAST_TABLE
        if primary == "xgboost"
        else DatabaseConfig.LGBM_FORECAST_TABLE
    )
    assert tables[DatabaseConfig.FORECAST_TABLE] is tables[expected]
    assert {row["run_type"] for row in records} == {"daily", "daily_lgbm"}
    assert records[-1]["status"] == "success"


def test_lgbm_failure_falls_back_and_both_fail_leave_main_untouched(
    monkeypatch, sample
):
    orders, as_of, _, xgb = sample
    monkeypatch.setattr(AppConfig, "PRIMARY_MODEL", "lgbm")
    monkeypatch.setattr(AppConfig, "RUN_LEGACY_MODEL", True)

    def fail(*args, **kwargs):
        raise RuntimeError("model failure")

    monkeypatch.setattr(model_routing, "forecast_lgbm_direct", fail)
    tables = {}
    records = []
    kwargs = dict(
        orders_loader=lambda: orders,
        save_table=lambda forecast, table: tables.__setitem__(table, forecast),
        read_previous=lambda table: None,
        log_run=lambda **row: records.append(row),
        as_of_date=as_of,
    )
    result = model_routing.run_model_routing(
        "daily", lambda: (xgb, {"run_type": "daily"}), **kwargs
    )
    assert result["status"] == "partial"
    assert result["main"] == "xgboost"
    assert DatabaseConfig.LGBM_FORECAST_TABLE not in tables
    assert "fallback" in records[-1]["error_message"]
    tables.clear()
    result = model_routing.run_model_routing("daily", fail, **kwargs)
    assert result["status"] == "failed"
    assert DatabaseConfig.FORECAST_TABLE not in tables


def test_sanity_failure_and_legacy_disabled(monkeypatch, sample):
    monkeypatch.setattr(
        model_routing,
        "check_lgbm_forecasts",
        lambda *args: {"finite_nonnegative": False},
    )
    result, tables, _ = setup_route(monkeypatch, sample, "lgbm")
    # setup_route uses the patched sanity function, so this is an XGBoost fallback.
    assert result["status"] == "partial"
    assert result["main"] == "xgboost"
    assert DatabaseConfig.LGBM_FORECAST_TABLE not in tables
    result, tables, records = setup_route(monkeypatch, sample, "lgbm", legacy=False)
    assert result["status"] == "failed"
    assert DatabaseConfig.FORECAST_TABLE not in tables
    assert [row["run_type"] for row in records] == ["daily_lgbm"]


def test_level_and_overlap_drift_checks(sample):
    orders, as_of, lgbm, _ = sample
    bundle = {"eligible_mlbs": list(lgbm)}
    healthy = model_routing.check_lgbm_forecasts(lgbm, bundle, orders, as_of_date=as_of)
    assert healthy["coverage_ok"] and healthy["level_ok"] and healthy["drift_ok"]
    mlb, (frame, _) = next(iter(lgbm.items()))
    previous = frame.iloc[:28].reset_index().rename(columns={"index": "date"})
    previous["mlb"] = mlb
    previous["prediction"] *= 3
    drift = model_routing.check_lgbm_forecasts(
        lgbm, bundle, orders, previous=previous, as_of_date=as_of
    )
    assert not drift["drift_ok"]


def test_legacy_disabled_publishes_only_lgbm(monkeypatch, sample):
    result, tables, records = setup_route(monkeypatch, sample, "lgbm", legacy=False)
    assert result["status"] == "success"
    assert result["main"] == "lgbm"
    assert set(tables) == {
        DatabaseConfig.LGBM_FORECAST_TABLE,
        DatabaseConfig.FORECAST_TABLE,
    }
    assert [row["run_type"] for row in records] == ["daily_lgbm"]


def test_daily_legacy_noop_keeps_existing_main(monkeypatch, sample):
    orders, as_of, lgbm, _ = sample
    monkeypatch.setattr(AppConfig, "PRIMARY_MODEL", "xgboost")
    monkeypatch.setattr(AppConfig, "RUN_LEGACY_MODEL", True)
    monkeypatch.setattr(
        model_routing,
        "forecast_lgbm_direct",
        lambda value, as_of_date=None: (
            lgbm,
            {"eligible_mlbs": list(lgbm), "path": "/tmp/model.joblib"},
        ),
    )
    tables = {}
    records = []
    result = model_routing.run_model_routing(
        "daily",
        lambda: ({}, {"run_type": "daily", "status": "success", "models_updated": 0}),
        orders_loader=lambda: orders,
        save_table=lambda forecast, table: tables.__setitem__(table, forecast),
        read_previous=lambda table: None,
        log_run=lambda **row: records.append(row),
        as_of_date=as_of,
    )
    assert result["status"] == "success"
    assert result["main"] == "unchanged"
    assert set(tables) == {DatabaseConfig.LGBM_FORECAST_TABLE}
    assert records[-1]["status"] == "success"
