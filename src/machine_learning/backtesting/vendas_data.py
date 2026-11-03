"""
Load Greycom's Mercado Livre invoice sales (NF-e series 2) for backtesting.

The daily file ``vendas_sku_dia_serie2.csv`` has one row per SKU and invoice-issue day
(Brasília local dates, no timezone; cancelled invoices already excluded; days without
sales absent). It is turned into pseudo-orders shaped like ``public.view_enrico``: one
order per SKU-day with gross units sold. Everything in the harness aggregates orders to
daily units and revenue per series, so one pseudo-order per day is equivalent to the
item-level invoice file for this purpose.

The series identifier is the SKU (there is no MLB column), and the target is gross
units sold (``unidades_vendidas``): returns are dated on the day goods came back, not
the day of the sale, so net units would misplace them.
"""

import logging
import re
from pathlib import Path
from typing import Iterable, Union

import pandas as pd

from backtesting.data_prep import load_orders
from backtesting.synthetic_data import DATA_DIR

logger = logging.getLogger(__name__)

VENDAS_CSV = DATA_DIR / "vendas_ml_serie2_2026-11-01" / "vendas_sku_dia_serie2.csv"
VENDAS_END_DATE = "2026-10-31"  # 1 Nov 2026 is incomplete in the extract

# World Cup sticker packets and albums: about 92% of 2026 units, excluded from training
# and scoring so they do not swamp the results
WORLD_CUP_SKUS = (
    "ENVELOPECOPA2026",
    "ENVELOPECOPA",
    "ALBUMCOPA2026",
    "ALBUMCORTESIA2026",
    "ALBUMCOPA2026CAPADURA",
    "ALBUMCOPABRINDE",
    "ALBUMCOPA",
    "ALBUM2026OURO",
    "ALBUM2026PRATA",
    "ALBUMCOPACAPADURA",
    "ALBUMFIFACLUBE25",
)
SUSPECT_PATTERN = re.compile(r"COPA|FIFA|ALBUM|FIGURINHA")

# LEIAME.md reference figures: authorised sales by invoice year (items sold, net revenue
# in R$). 2026 runs to 1 Nov inclusive, so the check uses the file before truncation.
REFERENCE_FIGURES = pd.DataFrame(
    {
        "itens_vendidos": [108, 74_100, 122_630, 313_100, 409_189, 932_439],
        "receita_liquida": [
            23_737,
            9_335_376,
            25_444_358,
            60_073_402,
            82_815_821,
            184_286_835,
        ],
    },
    index=pd.Index([2021, 2022, 2023, 2024, 2025, 2026], name="year"),
)


def read_vendas_daily(path: Union[str, Path] = VENDAS_CSV) -> pd.DataFrame:
    """Read the SKU-day file with SKUs as strings and dates as naive calendar days."""
    daily = pd.read_csv(path, dtype={"sku": str}, parse_dates=["data"])
    if daily["data"].dt.tz is not None:
        raise ValueError("Invoice dates must be naive calendar days")
    return daily


def reference_check(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Yearly totals of the daily file next to the LEIAME.md reference figures.

    The reference counts items sold (invoice lines), not units, so units are reported
    alongside but not compared.
    """
    totals = daily.groupby(daily["data"].dt.year.rename("year"))[
        ["unidades_vendidas", "itens_vendidos", "receita_liquida"]
    ].sum()
    check = totals.join(REFERENCE_FIGURES, rsuffix="_ref")
    check["itens_diff"] = check["itens_vendidos"] - check["itens_vendidos_ref"]
    check["receita_diff"] = check["receita_liquida"] - check["receita_liquida_ref"]
    return check


def suspect_skus(daily: pd.DataFrame, excluded: Iterable[str]) -> pd.DataFrame:
    """SKUs whose description looks World Cup related but that are not excluded."""
    description = daily["descricao"].fillna("").str.upper()
    hits = daily[
        description.str.contains(SUSPECT_PATTERN) & ~daily["sku"].isin(excluded)
    ]
    return (
        hits.groupby("sku")
        .agg(
            descricao=("descricao", "last"),
            units=("unidades_vendidas", "sum"),
            days=("data", "size"),
        )
        .sort_values("units", ascending=False)
    )


def load_vendas_orders(
    path: Union[str, Path, pd.DataFrame] = VENDAS_CSV,
    end_date: Union[str, pd.Timestamp] = VENDAS_END_DATE,
    exclude_skus: Iterable[str] = WORLD_CUP_SKUS,
) -> pd.DataFrame:
    """
    Pseudo-orders shaped like ``public.view_enrico``, normalised by ``load_orders``.

    One row per SKU-day with ``unidades_vendidas > 0``; the unit price is the
    quantity-weighted invoiced price ``valor_produtos / unidades_vendidas``. Rows after
    ``end_date`` and the ``exclude_skus`` are dropped. Dates stay on their calendar day.

    Args:
        path: The daily CSV, or a DataFrame already read with ``read_vendas_daily``.
        end_date: Last day kept.
        exclude_skus: SKUs removed from training and scoring.

    Returns:
        DataFrame with order_id, date_created, fulfilled, order_items_item_seller_sku,
        order_items_quantity, order_items_unit_price, mlb (= SKU) and date.
    """
    daily = path.copy() if isinstance(path, pd.DataFrame) else read_vendas_daily(path)
    daily["sku"] = daily["sku"].astype(str)
    exclude_skus = list(exclude_skus)

    for year, row in reference_check(daily).iterrows():
        logger.info(
            f"{year}: {row['unidades_vendidas']:,.0f} units, "
            f"{row['itens_vendidos']:,.0f} items (ref {row['itens_vendidos_ref']:,.0f}), "
            f"R$ {row['receita_liquida']:,.0f} (ref {row['receita_liquida_ref']:,.0f})"
        )

    end_date = pd.Timestamp(end_date).normalize()
    late = daily["data"] > end_date
    logger.info(
        f"Dropped {late.sum()} rows ({daily.loc[late, 'unidades_vendidas'].sum():,.0f} "
        f"units) after {end_date.date()}"
    )
    daily = daily[~late]

    excluded = daily["sku"].isin(exclude_skus)
    logger.info(
        f"Excluded {excluded.sum()} rows and "
        f"{daily.loc[excluded, 'unidades_vendidas'].sum():,.0f} units of "
        f"{daily['unidades_vendidas'].sum():,.0f} for {len(exclude_skus)} World Cup SKUs"
    )
    for sku, row in suspect_skus(daily, exclude_skus).iterrows():
        logger.warning(
            f"Not excluded but looks World Cup related: {sku} ({row['descricao']}), "
            f"{row['units']:,.0f} units on {row['days']} days"
        )
    daily = daily[~excluded & (daily["unidades_vendidas"] > 0)]

    if daily.duplicated(["sku", "data"]).any():
        raise ValueError("The daily file must have one row per SKU and day")
    daily = daily.sort_values(["data", "sku"], ignore_index=True)
    orders = pd.DataFrame(
        {
            "order_id": "NF2-" + daily.index.astype(str),
            "date_created": daily["data"],
            "fulfilled": True,
            "order_items_item_seller_sku": daily["sku"],
            "order_items_quantity": daily["unidades_vendidas"].astype(float),
            "order_items_unit_price": daily["valor_produtos"]
            / daily["unidades_vendidas"],
            "mlb": daily["sku"],
        }
    )
    orders = load_orders(orders)
    if not orders["date"].equals(daily["data"]):
        raise ValueError("Normalising orders shifted invoice dates")
    return orders
