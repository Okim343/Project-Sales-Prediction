"""Offline production adapter checks for dates, leakage, eligibility, and storage."""

import joblib
import numpy as np
import pandas as pd
import pytest

from backtesting.synthetic_data import generate_synthetic_orders
from config import AppConfig
from data_management.clean_sql_data import process_sales_data
from data_management.feature_creation import create_time_series_features
from estimation.eligibility import eligible_series
from estimation.lgbm_forecast import forecast_lgbm_direct, normalize_orders
from pipeline import model_routing


@pytest.fixture(scope="module")
def orders():
    return generate_synthetic_orders(
        n_skus=8, start="2024-01-01", end="2025-03-31", seed=23
    )[0]


def test_production_forecast_dates_output_bundle_and_eligibility(
    orders, monkeypatch, tmp_path
):
    monkeypatch.setattr(AppConfig, "LGBM_MODEL_DIR", tmp_path)
    as_of = pd.Timestamp("2025-04-01")
    forecasts, bundle = forecast_lgbm_direct(orders, as_of_date=as_of)
    assert bundle["cutoff"] == "2025-03-31"
    assert set(bundle["models"]) == {(1, 7), (8, 30), (31, 90)}
    assert bundle["config"]["n_estimators"] == 300
    assert bundle["config"]["calibration"] == "none"
    expected = pd.date_range("2025-04-02", periods=90)
    for frame, sku in forecasts.values():
        assert frame.index.equals(expected)
        assert frame.columns.tolist() == ["prediction"]
        values = frame.prediction.to_numpy()
        assert values.dtype.kind == "f"
        assert np.isfinite(values).all() and (values >= 0).all()
        assert sku is not None
    history = normalize_orders(orders)
    history = history[history.date <= pd.Timestamp("2025-03-31")]
    features = create_time_series_features(
        process_sales_data(history.drop(columns="date"))
    )
    assert set(forecasts) == set(
        eligible_series(features, pd.Timestamp("2025-03-31"), 90)
    )
    loaded = joblib.load(bundle["path"])
    assert loaded["config_hash"] == bundle["config_hash"]
    assert set(loaded["models"]) == set(bundle["models"])


def test_partial_today_and_future_orders_cannot_leak(orders, monkeypatch, tmp_path):
    monkeypatch.setattr(AppConfig, "LGBM_MODEL_DIR", tmp_path)
    baseline, _ = forecast_lgbm_direct(orders, as_of_date="2025-04-01")
    tampered = orders.copy()
    additions = tampered.iloc[:10].copy()
    additions["date_created"] = "2025-04-01T12:00:00"
    additions["order_items_quantity"] = 100_000
    additions["order_items_unit_price"] = 1_000_000
    tampered = pd.concat([tampered, additions], ignore_index=True)
    changed, _ = forecast_lgbm_direct(tampered, as_of_date="2025-04-01")
    assert set(changed) == set(baseline)
    for mlb in baseline:
        pd.testing.assert_frame_equal(baseline[mlb][0], changed[mlb][0])


def test_lgbm_import_window_uses_configured_months(monkeypatch):
    seen = {}
    monkeypatch.setattr(AppConfig, "LGBM_HISTORY_MONTHS", 21)

    def fake_import(**kwargs):
        seen.update(kwargs)
        return pd.DataFrame()

    monkeypatch.setattr(model_routing, "import_data_last_n_months", fake_import)
    model_routing.import_lgbm_history()
    assert seen["months"] == 21
