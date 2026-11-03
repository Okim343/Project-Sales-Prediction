"""Tests for the invoice-sales loader."""

import io

import pandas as pd
import pytest

from backtesting.vendas_data import (
    load_vendas_orders,
    read_vendas_daily,
    reference_check,
    suspect_skus,
)

DAILY_CSV = """data,sku,descricao,unidades_vendidas,itens_vendidos,receita_liquida,valor_produtos,unidades_liquidas
2025-12-30,100X150T27,0M,ETIQUETA 100X150,10.0000,2,120.00,100.00,10.0000
2025-12-30,AWR1-110X74,RIBBON CERA,4.0000,3,90.00,80.00,3.0000
2025-12-31,AWR1-110X74,RIBBON CERA,0.0000,0,0.00,0.00,-1.0000
2025-12-31,ENVELOPECOPA2026,COPA DO MUNDO 2026 - ENVELOPE,500.0000,9,3000.00,3000.00,500.0000
2025-12-31,ENVELOPEFIFACLUBE25,FIFA CLUB WORLD CUP 25 - ENVELOPE,7.0000,1,40.00,40.00,7.0000
2026-01-01,AWR1-110X74,RIBBON CERA,2.0000,1,50.00,50.00,2.0000
"""


def _daily():
    # The SKU 100X150T27,0M holds a comma, so quote it as a real export would
    text = DAILY_CSV.replace("100X150T27,0M", '"100X150T27,0M"')
    return read_vendas_daily(io.StringIO(text))


def test_read_keeps_sku_as_string_and_dates_naive():
    daily = _daily()
    assert daily["sku"].map(type).eq(str).all()
    assert "100X150T27,0M" in set(daily["sku"])
    assert daily["data"].dt.tz is None


def test_loader_columns_and_one_row_per_sku_day():
    orders = load_vendas_orders(_daily(), end_date="2026-01-31")
    assert list(orders.columns) == [
        "order_id",
        "date_created",
        "fulfilled",
        "order_items_item_seller_sku",
        "order_items_quantity",
        "order_items_unit_price",
        "mlb",
        "date",
    ]
    assert orders["order_id"].is_unique
    assert not orders.duplicated(["mlb", "date"]).any()
    assert orders["fulfilled"].all()
    assert (orders["mlb"] == orders["order_items_item_seller_sku"]).all()
    assert orders["order_items_quantity"].dtype == float


def test_unit_price_is_invoiced_value_over_units():
    orders = load_vendas_orders(_daily(), end_date="2026-01-31")
    ribbon = orders[orders["mlb"] == "AWR1-110X74"].set_index("date")
    # valor_produtos / unidades_vendidas, not receita_liquida
    assert ribbon.loc["2025-12-30", "order_items_unit_price"] == pytest.approx(20.0)
    assert ribbon.loc["2026-01-01", "order_items_unit_price"] == pytest.approx(25.0)


def test_zero_gross_rows_and_world_cup_skus_are_dropped():
    orders = load_vendas_orders(_daily(), end_date="2026-01-31")
    ribbon_dates = orders.loc[orders["mlb"] == "AWR1-110X74", "date"]
    assert pd.Timestamp("2025-12-31") not in set(ribbon_dates)
    assert "ENVELOPECOPA2026" not in set(orders["mlb"])
    # Flagged by description only, so kept
    assert "ENVELOPEFIFACLUBE25" in set(orders["mlb"])


def test_suspect_skus_reports_unlisted_matches():
    suspects = suspect_skus(_daily(), ["ENVELOPECOPA2026"])
    assert list(suspects.index) == ["ENVELOPEFIFACLUBE25"]


def test_dates_unchanged_and_end_date_respected():
    orders = load_vendas_orders(_daily(), end_date="2025-12-31")
    assert orders["date"].max() == pd.Timestamp("2025-12-31")
    assert set(orders["date"]) == {
        pd.Timestamp("2025-12-30"),
        pd.Timestamp("2025-12-31"),
    }
    # Midnight invoice dates stay on their calendar day after normalisation
    assert (orders["date_created"].dt.tz_localize(None) == orders["date"]).all()


def test_reference_check_counts_items_and_revenue_by_year():
    check = reference_check(_daily())
    assert check.loc[2025, "itens_vendidos_ref"] == 409_189
    assert check.loc[2025, "itens_vendidos"] == 15
    assert check.loc[2025, "receita_liquida"] == pytest.approx(3250.0)
    assert check.loc[2026, "unidades_vendidas"] == pytest.approx(2.0)
