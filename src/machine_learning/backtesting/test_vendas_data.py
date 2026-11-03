"""Tests for the invoice-sales loader, value-weighted WAPE and the --end-date option."""

import io

import numpy as np
import pandas as pd
import pytest

from backtesting.data_prep import last_unit_price
from backtesting.metrics import summarise, wape, wape_value
from backtesting.run_backtest import parse_args, truncate_and_cut, unit_concentration
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


def test_wape_value_by_hand():
    forecast = pd.Series([2.0, 0.0, 5.0])
    actual = pd.Series([1.0, 2.0, 5.0])
    weight = pd.Series([10.0, 1.0, 2.0])
    # w|e| = 10, 2, 0 -> 12 ; w*a = 10, 2, 10 -> 22
    assert wape_value(forecast, actual, weight) == pytest.approx(12 / 22)


def test_wape_value_equals_wape_with_equal_prices():
    rng = np.random.default_rng(0)
    forecast = pd.Series(rng.poisson(3, 50).astype(float))
    actual = pd.Series(rng.poisson(3, 50).astype(float))
    assert wape_value(forecast, actual, pd.Series(7.5, index=actual.index)) == (
        pytest.approx(wape(forecast, actual))
    )


def test_summarise_adds_wape_value_only_with_weights():
    results = pd.DataFrame(
        {
            "model": "m",
            "cutoff": pd.Timestamp("2025-01-01"),
            "mlb": ["a"] * 10 + ["b"] * 10,
            "h": list(range(1, 11)) * 2,
            "forecast": [1.0] * 10 + [4.0] * 10,
            "actual": 2.0,
            "scale": 1.0,
            "tier": "rest",
        }
    )
    assert "wape_value" not in summarise(results)
    weighted = summarise(
        results.assign(price_weight=[1.0] * 10 + [3.0] * 10), by_cutoff=True
    )
    overall = weighted[weighted["group_type"] == "overall"].iloc[0]
    # a: |e| 1 at price 1, b: |e| 2 at price 3 -> (10 + 60) / (20 + 60)
    assert overall["wape_value"] == pytest.approx(70 / 80)
    assert overall["wape"] == pytest.approx(30 / 40)
    assert list(weighted.loc[weighted["group_type"] == "cutoff", "group"]) == [
        "2025-01-01"
    ]


def test_price_weight_uses_last_price_at_cutoff_not_later():
    orders = load_vendas_orders(_daily(), end_date="2026-01-31")
    prices = last_unit_price(orders, pd.Timestamp("2025-12-31"))
    # The ribbon sold at 20 on 30 Dec and at 25 on 1 Jan; the cutoff price is 20
    assert prices["AWR1-110X74"] == pytest.approx(20.0)
    assert prices["100X150T27,0M"] == pytest.approx(10.0)
    assert last_unit_price(orders, pd.Timestamp("2026-01-01"))[
        "AWR1-110X74"
    ] == pytest.approx(25.0)


def _orders_until(end):
    dates = pd.date_range("2023-01-01", end, freq="D")
    return pd.DataFrame({"date": dates, "mlb": "a", "order_items_quantity": 1.0})


@pytest.mark.parametrize(
    "end_date, expected",
    [
        ("2025-12-31", ["2025-07-04", "2025-08-03", "2025-09-02", "2025-10-02"]),
        ("2026-10-31", ["2026-05-04", "2026-06-03", "2026-07-03", "2026-08-02"]),
    ],
)
def test_end_date_truncates_before_cutoffs(end_date, expected):
    args = parse_args(["--source", "vendas", "--end-date", end_date])
    orders, cutoffs = truncate_and_cut(_orders_until("2026-10-31"), args)
    assert [c.date().isoformat() for c in cutoffs] == expected
    assert orders["date"].max() == pd.Timestamp(end_date)


def test_end_date_sets_last_cutoff_even_without_sales_on_last_day():
    args = parse_args(["--end-date", "2025-12-31"])
    _, cutoffs = truncate_and_cut(_orders_until("2025-12-20"), args)
    assert cutoffs[-1] == pd.Timestamp("2025-10-02")


def test_without_end_date_cutoffs_follow_the_data():
    args = parse_args([])
    _, cutoffs = truncate_and_cut(_orders_until("2025-03-16"), args)
    assert cutoffs[-1] == pd.Timestamp("2024-12-16")


def test_unit_concentration_shares_use_one_model():
    rows = pd.DataFrame(
        {
            "model": ["m1"] * 3 + ["m2"] * 3,
            "mlb": ["a", "b", "c"] * 2,
            "actual": [6.0, 3.0, 1.0] * 2,
            "price_weight": [1.0, 2.0, 4.0] * 2,
        }
    )
    shares = unit_concentration(rows, top=2)
    assert list(shares["mlb"]) == ["a", "b"]
    assert list(shares["unit_share"]) == pytest.approx([0.6, 0.3])
    # value 6, 6, 4 of 16
    assert list(shares["value_share"]) == pytest.approx([6 / 16, 6 / 16])
