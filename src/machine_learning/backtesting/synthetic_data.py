"""
Synthetic order data shaped like the production view ``public.view_enrico``.

Daily demand per listing (MLB) is drawn from a negative binomial whose mean combines a
base level, growth trend, day-of-week profile, yearly seasonality, Black Friday and
Christmas effects, and price discounts. Stockout spells then censor observed sales to
zero, and listings launch and discontinue at staggered dates. The daily units are split
into individual order rows with the same columns as the view.

The defaults are calibrated to the historical export in ``data/raw_sql.csv`` (see
``calibration_report``). Every planted parameter is returned as ground truth so that
models can later be scored against true demand as well as observed sales.

Run ``python backtesting/synthetic_data.py`` to write ``data/synthetic_orders.csv`` plus
the ground-truth files and print the calibration report.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"

VIEW_COLUMNS = [
    "order_id",
    "date_created",
    "fulfilled",
    "order_items_item_seller_sku",
    "order_items_quantity",
    "order_items_unit_price",
    "mlb",
]

# Monday..Sunday demand multipliers measured on the historical export
DAY_OF_WEEK_PROFILE = np.array([1.22, 1.24, 1.23, 1.20, 1.07, 0.57, 0.46])

# Calibration constants (tuned so calibration_report matches the historical export)
BASE_LEVEL_MEDIAN = 0.30  # Median daily units at the start of a listing's life
BASE_LEVEL_SIGMA = 2.1  # Log-scale spread of listing volumes
MAX_BASE_LEVEL = 90.0
TREND_MEAN = 0.15  # Mean log-growth per year
TREND_SD = 0.45
NB_DISPERSION = 2.5  # Negative binomial shape; lower means burstier sales
LEVEL_PERSISTENCE = 0.98  # AR(1) coefficient of the slowly wandering log demand level
LEVEL_SD = 0.5  # Stationary standard deviation of that log level
LAUNCH_SHARE = 0.55  # Share of listings launched after the start date
DISCONTINUE_SHARE = 0.12
SECOND_LISTING_SHARE = 0.10  # Share of SKUs sold through two MLBs
UNFULFILLED_SHARE = 0.055
NEW_ORDER_PROB = 0.75  # Chance each extra unit on a day starts a new order

logger = logging.getLogger(__name__)


def generate_synthetic_orders(
    n_skus: int = 450,
    start: str = "2023-01-01",
    end: str = "2025-12-31",
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Generate synthetic order rows with known demand patterns.

    Args:
        n_skus: Number of SKUs. About 10% get a second MLB, so there are slightly more
            series than SKUs.
        start: First calendar date (inclusive).
        end: Last calendar date (inclusive).
        seed: Random seed; the same seed always gives identical output.

    Returns:
        Tuple of (orders, truth). ``orders`` has the columns of the production view.
        ``truth`` holds ``params`` (one row per MLB with the planted parameters) and
        ``daily`` (one row per MLB-day with true demand, observed sales, price,
        discount, and stockout flag).
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    n_days = len(dates)

    params = _draw_series_params(rng, n_skus, dates)
    n_series = len(params)

    calendar = _calendar_multipliers(dates)
    years = (np.arange(n_days) / 365.25)[None, :]

    yearly = np.exp(
        params["season_amp"].to_numpy()[:, None]
        * np.sin(2 * np.pi * (years - params["season_phase"].to_numpy()[:, None]))
    )
    trend = np.exp(params["trend"].to_numpy()[:, None] * years)

    discount = _draw_episodes(rng, n_series, n_days, mean_gap=60, min_len=3, max_len=10)
    discount_depth = rng.uniform(0.10, 0.30, size=(n_series, n_days)) * discount
    price_effect = (1 - discount_depth) ** (-params["elasticity"].to_numpy()[:, None])

    level_shocks = _ar1_log_level(rng, n_series, n_days)

    demand_mean = (
        params["base_level"].to_numpy()[:, None]
        * level_shocks
        * trend
        * yearly
        * calendar[None, :]
        * price_effect
    )

    alive = _lifetime_mask(params, dates)
    demand_mean = np.where(alive, demand_mean, 0.0)

    gamma = rng.gamma(NB_DISPERSION, demand_mean / NB_DISPERSION)
    true_demand = rng.poisson(gamma)

    stockout = _draw_episodes(
        rng, n_series, n_days, mean_gap=120, min_len=3, max_len=25
    )
    stockout &= alive
    observed = np.where(stockout, 0, true_demand)

    unit_price = params["base_price"].to_numpy()[:, None] * (1 - discount_depth)

    daily = _build_daily_frame(
        params,
        dates,
        alive,
        demand_mean,
        true_demand,
        observed,
        unit_price,
        discount_depth,
        stockout,
    )
    orders = _explode_to_orders(rng, daily)

    logger.info(
        f"Generated {len(orders):,} orders for {n_series} MLBs "
        f"({params['sku'].nunique()} SKUs) from {dates[0].date()} to {dates[-1].date()}"
    )
    return orders, {"params": params, "daily": daily}


def save_synthetic(
    orders: pd.DataFrame, truth: Dict[str, pd.DataFrame], directory: Path = DATA_DIR
) -> Path:
    """Write orders as CSV and ground truth as parquet/CSV; return the orders path."""
    directory.mkdir(parents=True, exist_ok=True)
    orders_path = directory / "synthetic_orders.csv"
    orders.to_csv(orders_path, index=False)
    truth["daily"].to_parquet(directory / "synthetic_truth.parquet", index=False)
    truth["params"].to_csv(directory / "synthetic_params.csv", index=False)
    return orders_path


def calibration_report(orders: pd.DataFrame) -> Dict[str, float]:
    """
    Summary statistics used to calibrate the generator against real data.

    Statistics are computed on a SKU x day grid spanning the whole date range, with
    missing days counted as zero, so real and synthetic exports are measured the same
    way.
    """
    data = orders.copy()
    data["date"] = (
        pd.to_datetime(data["date_created"], utc=True, format="mixed")
        .dt.tz_localize(None)
        .dt.normalize()
    )
    grid = (
        data.groupby(["order_items_item_seller_sku", "date"])["order_items_quantity"]
        .sum()
        .unstack(fill_value=0)
    )
    grid = grid.reindex(
        columns=pd.date_range(grid.columns.min(), grid.columns.max()), fill_value=0
    )
    totals = grid.sum(axis=1).sort_values(ascending=False)
    top = grid.loc[totals.index[:100]]

    daily = grid.sum(axis=0)
    weekday_ratio = daily.groupby(daily.index.dayofweek).mean() / daily.mean()

    black_friday_ratios = []
    for year in sorted(set(daily.index.year)):
        week_start = _black_friday(year) - pd.Timedelta(days=4)
        week = daily[week_start : week_start + pd.Timedelta(days=6)]
        reference = daily[
            week_start - pd.Timedelta(days=56) : week_start - pd.Timedelta(days=1)
        ]
        if len(week) == 7 and len(reference) == 56:
            black_friday_ratios.append(week.sum() / (reference.mean() * 7))

    monthly = daily.resample("ME").sum()
    full_months = monthly.iloc[1:-1] if len(monthly) > 2 else monthly

    return {
        "n_skus": int(grid.shape[0]),
        "n_days": int(grid.shape[1]),
        "zero_share_all": float((grid == 0).to_numpy().mean()),
        "zero_share_top100": float((top == 0).to_numpy().mean()),
        "median_sku_daily_units": float(grid.mean(axis=1).median()),
        "median_top100_daily_units": float(top.mean(axis=1).median()),
        **{
            f"weekday_ratio_{d}": float(weekday_ratio.get(i, np.nan))
            for i, d in enumerate(["mon", "tue", "wed", "thu", "fri", "sat", "sun"])
        },
        "black_friday_week_vs_prior_8w": float(np.mean(black_friday_ratios))
        if black_friday_ratios
        else float("nan"),
        "first_full_month_units": float(full_months.iloc[0]),
        "last_full_month_units": float(full_months.iloc[-1]),
        "unfulfilled_share": float(
            (~data["fulfilled"].astype(str).str.lower().eq("true")).mean()
        ),
        "median_unit_price": float(data["order_items_unit_price"].median()),
    }


def _draw_series_params(
    rng: np.random.Generator, n_skus: int, dates: pd.DatetimeIndex
) -> pd.DataFrame:
    """Draw per-series parameters; some SKUs are sold through two MLBs."""
    sku_ids = [f"SYN{i:04d}" for i in range(1, n_skus + 1)]
    second = rng.random(n_skus) < SECOND_LISTING_SHARE
    skus = sku_ids + [s for s, extra in zip(sku_ids, second) if extra]
    n_series = len(skus)

    mlb_numbers = rng.choice(np.arange(10**9, 10**10), size=n_series, replace=False)
    base_level = np.minimum(
        BASE_LEVEL_MEDIAN * np.exp(rng.normal(0, BASE_LEVEL_SIGMA, n_series)),
        MAX_BASE_LEVEL,
    )
    base_price_by_sku = dict(
        zip(sku_ids, np.clip(115 * np.exp(rng.normal(0, 0.9, n_skus)), 7, 6650))
    )

    span = (dates[-1] - dates[0]).days
    launched_late = rng.random(n_series) < LAUNCH_SHARE
    launch_offset = np.where(launched_late, rng.integers(0, span - 60, n_series), 0)
    discontinued = rng.random(n_series) < DISCONTINUE_SHARE
    end_offset = np.where(
        discontinued,
        np.minimum(span, launch_offset + rng.integers(120, max(span, 121), n_series)),
        span,
    )

    return pd.DataFrame(
        {
            "mlb": [f"MLB{n}" for n in mlb_numbers],
            "sku": skus,
            "base_level": base_level,
            "trend": rng.normal(TREND_MEAN, TREND_SD, n_series),
            "season_amp": rng.uniform(0.0, 0.25, n_series),
            "season_phase": rng.uniform(0.0, 1.0, n_series),
            "elasticity": rng.uniform(1.0, 3.0, n_series),
            "base_price": [base_price_by_sku[s] for s in skus],
            "launch_date": dates[0] + pd.to_timedelta(launch_offset, unit="D"),
            "end_date": dates[0] + pd.to_timedelta(end_offset, unit="D"),
        }
    )


def _black_friday(year: int) -> pd.Timestamp:
    """Fourth Friday of November."""
    first = pd.Timestamp(year=year, month=11, day=1)
    first_friday = first + pd.Timedelta(days=(4 - first.dayofweek) % 7)
    return first_friday + pd.Timedelta(weeks=3)


def _calendar_multipliers(dates: pd.DatetimeIndex) -> np.ndarray:
    """Day-of-week profile plus Black Friday and Christmas effects shared by all series."""
    multiplier = DAY_OF_WEEK_PROFILE[dates.dayofweek].copy()
    for year in sorted(set(dates.year)):
        black_friday = _black_friday(year)
        week = (dates >= black_friday - pd.Timedelta(days=4)) & (
            dates <= black_friday + pd.Timedelta(days=3)
        )
        peak = (dates >= black_friday) & (dates <= black_friday + pd.Timedelta(days=3))
        multiplier[week] *= 1.08
        multiplier[peak] *= 1.2
        christmas_run_up = (dates >= f"{year}-12-01") & (dates <= f"{year}-12-20")
        holidays = (dates >= f"{year}-12-24") & (dates <= f"{year}-12-26")
        multiplier[christmas_run_up] *= 1.15
        multiplier[holidays] *= 0.5
    return multiplier


def _draw_episodes(
    rng: np.random.Generator,
    n_series: int,
    n_days: int,
    mean_gap: int,
    min_len: int,
    max_len: int,
) -> np.ndarray:
    """Boolean (series x day) mask of randomly placed episodes."""
    mask = np.zeros((n_series, n_days), dtype=bool)
    for i in range(n_series):
        day = int(rng.exponential(mean_gap))
        while day < n_days:
            length = int(rng.integers(min_len, max_len + 1))
            mask[i, day : day + length] = True
            day += length + int(rng.exponential(mean_gap))
    return mask


def _ar1_log_level(rng: np.random.Generator, n_series: int, n_days: int) -> np.ndarray:
    """Persistent multiplicative demand shocks: exp of a mean-zero AR(1) process."""
    innovation_sd = LEVEL_SD * np.sqrt(1 - LEVEL_PERSISTENCE**2)
    shocks = rng.normal(0, innovation_sd, (n_series, n_days))
    level = np.empty((n_series, n_days))
    level[:, 0] = rng.normal(0, LEVEL_SD, n_series)
    for t in range(1, n_days):
        level[:, t] = LEVEL_PERSISTENCE * level[:, t - 1] + shocks[:, t]
    return np.exp(level - LEVEL_SD**2 / 2)


def _lifetime_mask(params: pd.DataFrame, dates: pd.DatetimeIndex) -> np.ndarray:
    """True on days between each series' launch and end date."""
    date_values = dates.to_numpy()[None, :]
    return (date_values >= params["launch_date"].to_numpy()[:, None]) & (
        date_values <= params["end_date"].to_numpy()[:, None]
    )


def _build_daily_frame(
    params,
    dates,
    alive,
    demand_mean,
    true_demand,
    observed,
    unit_price,
    discount_depth,
    stockout,
) -> pd.DataFrame:
    """Long-format ground truth restricted to each series' lifetime."""
    series_idx, day_idx = np.nonzero(alive)
    return pd.DataFrame(
        {
            "mlb": params["mlb"].to_numpy()[series_idx],
            "sku": params["sku"].to_numpy()[series_idx],
            "date": dates[day_idx],
            "demand_mean": demand_mean[series_idx, day_idx],
            "true_demand": true_demand[series_idx, day_idx],
            "observed": observed[series_idx, day_idx],
            "unit_price": unit_price[series_idx, day_idx].round(2),
            "discount": discount_depth[series_idx, day_idx].round(3),
            "stockout": stockout[series_idx, day_idx],
        }
    )


def _explode_to_orders(rng: np.random.Generator, daily: pd.DataFrame) -> pd.DataFrame:
    """Split each series-day's observed units into individual order rows."""
    sold = daily[daily["observed"] > 0].reset_index(drop=True)
    units = sold["observed"].to_numpy()
    cell = np.repeat(np.arange(len(sold)), units)

    first_unit = np.r_[True, cell[1:] != cell[:-1]]
    starts_order = first_unit | (rng.random(len(cell)) < NEW_ORDER_PROB)
    order_index = np.cumsum(starts_order) - 1
    quantity = np.bincount(order_index)
    order_cell = cell[starts_order]

    seconds = rng.integers(0, 86400, len(order_cell))
    timestamps = sold["date"].to_numpy()[order_cell] + seconds.astype("timedelta64[s]")
    price_noise = rng.normal(1.0, 0.01, len(order_cell))

    orders = pd.DataFrame(
        {
            "date_created": pd.to_datetime(timestamps),
            "fulfilled": rng.random(len(order_cell)) >= UNFULFILLED_SHARE,
            "order_items_item_seller_sku": sold["sku"].to_numpy()[order_cell],
            "order_items_quantity": quantity,
            "order_items_unit_price": np.maximum(
                (sold["unit_price"].to_numpy()[order_cell] * price_noise).round(2), 1.0
            ),
            "mlb": sold["mlb"].to_numpy()[order_cell],
        }
    ).sort_values("date_created", ignore_index=True)

    orders.insert(0, "order_id", 2_000_000_000_000_000 + np.arange(len(orders)))
    orders["date_created"] = orders["date_created"].dt.strftime(
        "%Y-%m-%d %H:%M:%S+00:00"
    )
    return orders[VIEW_COLUMNS]


def main():
    """Generate, save, and compare synthetic data with the historical export."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    orders, truth = generate_synthetic_orders()
    path = save_synthetic(orders, truth)
    print(f"Saved {len(orders):,} synthetic orders to {path}")

    synthetic = calibration_report(orders)
    real_path = DATA_DIR / "raw_sql.csv"
    real = (
        calibration_report(pd.read_csv(real_path, low_memory=False))
        if real_path.exists()
        else {}
    )

    print(f"\n{'statistic':34s} {'synthetic':>14s} {'real':>14s}")
    for key, value in synthetic.items():
        real_value = real.get(key, float("nan"))
        print(f"{key:34s} {value:14.3f} {real_value:14.3f}")


if __name__ == "__main__":
    sys.exit(main())
