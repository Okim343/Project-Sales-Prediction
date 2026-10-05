"""Smoke test for the production-model backtest adapter."""

import pandas as pd

from backtesting.current_model import forecast_current_model
from backtesting.data_prep import build_training_features, load_orders
from backtesting.synthetic_data import generate_synthetic_orders


def test_production_model_forecasts_from_cutoff():
    orders, _ = generate_synthetic_orders(
        n_skus=30, start="2024-01-01", end="2025-03-31", seed=5
    )
    orders = load_orders(orders)
    cutoff = pd.Timestamp("2025-01-31")
    features = build_training_features(orders, cutoff)
    busiest = features.groupby("mlb")["quant"].sum().nlargest(2).index.tolist()

    forecasts = forecast_current_model(features, busiest, cutoff, horizon=14, n_jobs=2)

    assert set(forecasts) == set(busiest)
    for forecast in forecasts.values():
        assert forecast.index[0] == cutoff + pd.Timedelta(days=1)
        assert len(forecast) == 14
        assert (forecast >= 0).all()
