"""Offline checks that staged forecast writes preserve a previous main table."""

import pandas as pd
import pytest
from sqlalchemy import create_engine, text

from database_utils import DatabaseManager


def test_atomic_save_and_failed_stage_preserves_previous_table(monkeypatch):
    manager = DatabaseManager()
    manager._engine = create_engine("sqlite:///:memory:")
    dates = pd.date_range("2025-01-01", periods=2)
    forecasts = {"MLB": (pd.DataFrame({"prediction": [1.1, 2.2]}, index=dates), "SKU")}
    manager.save_forecasts_atomic(forecasts, "forecast_main")
    with manager.engine.connect() as conn:
        before = conn.execute(
            text("SELECT prediction FROM forecast_main ORDER BY date")
        ).fetchall()
    assert [row[0] for row in before] == [1.1, 2.2]

    def fail_stage(*args, **kwargs):
        raise RuntimeError("staging failed")

    monkeypatch.setattr(pd.DataFrame, "to_sql", fail_stage)
    with pytest.raises(RuntimeError, match="staging failed"):
        manager.save_forecasts_atomic(forecasts, "forecast_main")
    with manager.engine.connect() as conn:
        after = conn.execute(
            text("SELECT prediction FROM forecast_main ORDER BY date")
        ).fetchall()
    assert after == before


def test_atomic_save_refuses_empty_or_unsafe_names():
    manager = DatabaseManager()
    with pytest.raises(ValueError):
        manager.save_forecasts_atomic({}, "forecast_main")
    with pytest.raises(ValueError):
        manager.save_forecasts_atomic(
            {"MLB": (pd.DataFrame({"prediction": [1]}), None)}, "main;DROP"
        )
