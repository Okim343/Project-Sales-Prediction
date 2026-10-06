"""
Global LightGBM challenger for the backtest harness.

One LightGBM model with a Tweedie objective is trained on every series up to the cutoff
(the production model instead fits one XGBoost per listing). Features describe the
target date (weekday, holidays, Black Friday, Christmas) as well as each series' recent
sales, price and volume level.

Two forecasting strategies are available:

- **recursive**: a one-step-ahead model whose predictions are fed back in as lags;
- **direct**: one model per horizon block (days 1-7, 8-30, 31-90 by default). Features
  are anchored at the forecast origin, plus target-date lags that are known at the
  origin for every horizon in the block (lag k is used only if k >= the block's last
  day), so no block ever sees a value from after its origin.

Everything is computed from a daily grid built from orders up to the cutoff only:
units per series from its first sale to the cutoff (zero-filled) and the unit price
(daily revenue / units, carried forward on days without sales). Future prices are
unknown at the cutoff, so forecasts assume the last known price.

Stockout handling (``stockout="mask"``) is a proxy for the missing stock data: a
zero-sales run is treated as a stockout when the series' recent rate makes that many
consecutive zeros implausible. Flagged days are dropped as training targets and
treated as missing in lag and rolling features.
"""

import logging
import os
import warnings
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple

import holidays
import lightgbm as lgb
import numpy as np
import pandas as pd

from backtesting.folds import horizon_dates

logger = logging.getLogger(__name__)

FEATURE_GROUPS = ("calendar", "lags", "price", "holidays", "volume")
CATEGORICAL = ["series"]

RECURSIVE_LAGS = (1, 7, 14, 28)
DIRECT_TARGET_LAGS = (7, 14, 28)
LAG_YEAR = 364
AGE_CAP = 180


@dataclass(frozen=True)
class LGBMConfig:
    """Settings for one LightGBM challenger variant."""

    strategy: str = "direct"  # "direct" or "recursive"
    feature_groups: Tuple[str, ...] = FEATURE_GROUPS
    use_lag_364: bool = True
    use_series_id: bool = True
    # Individual features to leave out. Month is dropped by default: with a year or
    # less of history it mostly encodes the growth trend and is extrapolated to months
    # never seen in training, which biased forecasts in the backtest.
    drop_columns: Tuple[str, ...] = ("month",)
    tweedie_power: float = 1.5
    weight_power: float = (
        0.0  # Sample weight = (series' 28-day mean at anchor) ** power
    )
    calibration: str = "none"  # none, overall, block, tier, block_tier, power
    calibration_shrink: float = 1.0  # fraction of validation ratio's move from 1
    calibration_floor: float = 0.8  # minimum applied factor after shrinkage
    target_scale: int = 0  # 0, 28, or 91: fit sales / floored origin level
    stockout: str = "mask"  # "mask" or "none"
    blocks: Tuple[Tuple[int, int], ...] = ((1, 7), (8, 30), (31, 90))
    origin_steps: Tuple[int, ...] = (1, 3, 7)  # Spacing of training origins per block
    train_days: int = 365  # Days of training targets before the cutoff
    n_estimators: int = 300
    learning_rate: float = 0.05
    num_leaves: int = 63
    min_child_samples: int = 100
    seed: int = 42
    params: Dict = field(default_factory=dict, compare=False, hash=False)


# Panel ---------------------------------------------------------------------------------


@dataclass
class Panel:
    """Wide daily grid for every series that sold at least once up to the cutoff."""

    mlbs: np.ndarray  # (S,)
    dates: pd.DatetimeIndex  # (T,), ends on the cutoff
    sales: np.ndarray  # (S, T) units; NaN before the first sale
    price: np.ndarray  # (S, T) last known unit price; NaN before the first sale
    first: np.ndarray  # (S,) column of the first sale


def build_panel(orders: pd.DataFrame, cutoff: pd.Timestamp) -> Panel:
    """Daily units and unit price per series from orders dated on or before the cutoff."""
    cutoff = pd.Timestamp(cutoff).normalize()
    data = orders[orders["date"] <= cutoff]
    revenue = data["order_items_quantity"] * data["order_items_unit_price"]
    daily = (
        data.assign(revenue=revenue)
        .groupby(["mlb", "date"])[["order_items_quantity", "revenue"]]
        .sum()
    )
    dates = pd.date_range(data["date"].min(), cutoff, freq="D")
    units = daily["order_items_quantity"].unstack().reindex(columns=dates)
    revenue = daily["revenue"].unstack().reindex(columns=dates)

    sold = units.fillna(0).to_numpy(dtype=float) > 0
    first = sold.argmax(axis=1)
    started = np.arange(len(dates))[None, :] >= first[:, None]

    sales = np.where(started, units.fillna(0).to_numpy(dtype=float), np.nan)
    unit_price = (revenue / units.where(units > 0)).ffill(axis=1).to_numpy(dtype=float)
    unit_price = np.where(started, unit_price, np.nan)

    if dates[-1] != cutoff or data["date"].max() > cutoff:
        raise ValueError("Panel contains dates after the cutoff")
    return Panel(units.index.to_numpy(), dates, sales, unit_price, first)


def flag_stockouts(
    sales: np.ndarray,
    min_run: int = 3,
    max_run: int = 30,
    min_expected_units: float = 4.0,
    rate_window: int = 28,
) -> np.ndarray:
    """
    Boolean mask of zero-sales days that look like stockouts.

    A run of consecutive zero days is flagged when

    - it lasts ``min_run`` to ``max_run`` days and sales resumed afterwards (longer or
      still-open runs are more often delistings or the end of a product's life), and
    - the series' mean daily sales over the ``rate_window`` days before the run, times
      the run length, is at least ``min_expected_units`` (under a Poisson rate that many
      zeros in a row would have probability below exp(-min_expected_units)).

    Only data inside the given array is used, so flags at a cutoff never depend on
    later sales. On the synthetic data (true stockouts known) the defaults flag about as
    many days as are truly out of stock, with roughly 50% precision and recall.
    """
    mask = np.zeros(sales.shape, dtype=bool)
    for s in range(sales.shape[0]):
        row = sales[s]
        zero = row == 0
        if not zero.any():
            continue
        edges = np.diff(np.concatenate([[0], zero.astype(int), [0]]))
        for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            length = end - start
            if not min_run <= length <= max_run or end >= len(row):
                continue
            before = row[max(0, start - rate_window) : start]
            before = before[~np.isnan(before)]
            if len(before) and before.mean() * length >= min_expected_units:
                mask[s, start:end] = True
    return mask


def _trailing_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Mean of the last ``window`` columns up to and including each column, NaN-aware."""
    filled = np.nan_to_num(values)
    present = (~np.isnan(values)).astype(float)
    pad = np.zeros((values.shape[0], 1))
    total = np.cumsum(np.hstack([pad, filled]), axis=1)
    count = np.cumsum(np.hstack([pad, present]), axis=1)
    sums = total[:, window:] - total[:, :-window]
    counts = count[:, window:] - count[:, :-window]
    head_sums = total[:, 1:window]
    head_counts = count[:, 1:window]
    sums = np.hstack([head_sums, sums])[:, : values.shape[1]]
    counts = np.hstack([head_counts, counts])[:, : values.shape[1]]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(counts > 0, sums / counts, np.nan)


def _trailing_max(values: np.ndarray, window: int) -> np.ndarray:
    """Max of the last ``window`` columns up to and including each column."""
    frame = pd.DataFrame(values.T)
    return frame.rolling(window, min_periods=1).max().to_numpy().T


def _take(matrix: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """matrix[rows, cols] with NaN where the column index is out of range."""
    out = matrix[rows, np.clip(cols, 0, matrix.shape[1] - 1)].astype(float)
    out[(cols < 0) | (cols >= matrix.shape[1])] = np.nan
    return out


# Calendar ------------------------------------------------------------------------------


def black_friday(year: int) -> pd.Timestamp:
    """The day after the fourth Thursday of November (US Thanksgiving)."""
    first = pd.Timestamp(year=year, month=11, day=1)
    return first + pd.Timedelta(days=(3 - first.dayofweek) % 7 + 22)


def calendar_features(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Calendar and holiday features for each date, in date order."""
    years = sorted(set(dates.year))
    brazil = holidays.Brazil(years=years)
    frame = pd.DataFrame(
        {
            "dow": dates.dayofweek,
            "dom": dates.day,
            "month": dates.month,
            "br_holiday": [d in brazil for d in dates.date],
        }
    )
    bf_week = np.zeros(len(dates), dtype=bool)
    bf_peak = np.zeros(len(dates), dtype=bool)
    for year in years:
        friday = black_friday(year)
        # Monday before Black Friday through Cyber Monday; peak is Friday-Monday
        bf_week |= (dates >= friday - pd.Timedelta(days=4)) & (
            dates <= friday + pd.Timedelta(days=3)
        )
        bf_peak |= (dates >= friday) & (dates <= friday + pd.Timedelta(days=3))
    frame["bf_week"] = bf_week
    frame["bf_peak"] = bf_peak
    frame["xmas_runup"] = (dates.month == 12) & (dates.day <= 20)
    frame["xmas_hol"] = (dates.month == 12) & dates.day.isin([24, 25, 26, 31])
    return frame.astype(float)


GROUP_COLUMNS = {
    "calendar": ["dow", "dom", "month"],
    "holidays": ["br_holiday", "bf_week", "bf_peak", "xmas_runup", "xmas_hol"],
    "price": ["price", "price_rel_28", "price_rel_max_90"],
    "volume": ["rmean_91", "age", "series"],
}


def feature_columns(
    config: LGBMConfig, block: Optional[Tuple[int, int]] = None
) -> List[str]:
    """Columns used by a variant (for the direct strategy, by one horizon block)."""
    columns = ["h"] if config.strategy == "direct" else []
    for group in config.feature_groups:
        if group == "lags":
            if config.strategy == "recursive":
                lags = [f"lag_{k}" for k in RECURSIVE_LAGS]
                columns += lags + ["rmean_7", "rmean_28"]
            else:
                valid = [f"lag_{k}" for k in DIRECT_TARGET_LAGS if k >= block[1]]
                columns += ["last_1", "rmean_7", "rmean_28", "wd_mean_4w", "lag_sw"]
                columns += valid
            if config.use_lag_364:
                columns.append(f"lag_{LAG_YEAR}")
        else:
            columns += GROUP_COLUMNS[group]
    dropped = set(config.drop_columns) | (set() if config.use_series_id else {"series"})
    return [c for c in columns if c not in dropped]


# Feature builders ----------------------------------------------------------------------


@dataclass
class _Matrices:
    """Per-column feature inputs derived from the sales and price grids."""

    sales: np.ndarray  # Sales with stockout days set to NaN (if masking)
    price: np.ndarray
    rmean_7: np.ndarray
    rmean_28: np.ndarray
    rmean_91: np.ndarray
    price_mean_28: np.ndarray
    price_max_90: np.ndarray

    @classmethod
    def from_grids(cls, sales: np.ndarray, price: np.ndarray) -> "_Matrices":
        return cls(
            sales,
            price,
            _trailing_mean(sales, 7),
            _trailing_mean(sales, 28),
            _trailing_mean(sales, 91),
            _trailing_mean(price, 28),
            _trailing_max(price, 90),
        )


def _common_features(
    m: _Matrices,
    first: np.ndarray,
    cal: pd.DataFrame,
    s: np.ndarray,
    anchor: np.ndarray,
    target: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Features shared by both strategies; ``anchor`` is the last known column."""
    price = _take(m.price, s, anchor)
    with np.errstate(invalid="ignore", divide="ignore"):
        rel_28 = price / _take(m.price_mean_28, s, anchor) - 1
        rel_max = price / _take(m.price_max_90, s, anchor) - 1
    out = {
        "rmean_7": _take(m.rmean_7, s, anchor),
        "rmean_28": _take(m.rmean_28, s, anchor),
        "rmean_91": _take(m.rmean_91, s, anchor),
        "price": price,
        "price_rel_28": rel_28,
        "price_rel_max_90": rel_max,
        # Capped so forecasts never extrapolate past ages seen in training; still
        # captures the ramp-up after a launch
        "age": np.minimum(anchor - first[s], AGE_CAP).astype(float),
        f"lag_{LAG_YEAR}": _take(m.sales, s, target - LAG_YEAR),
        "series": s,
    }
    for column in cal.columns:
        out[column] = cal[column].to_numpy()[target]
    return out


def _recursive_frame(
    m: _Matrices, first: np.ndarray, cal: pd.DataFrame, s: np.ndarray, t: np.ndarray
) -> pd.DataFrame:
    """One-step features for target columns ``t`` (lags end at t - 1)."""
    out = _common_features(m, first, cal, s, t - 1, t)
    for k in RECURSIVE_LAGS:
        out[f"lag_{k}"] = _take(m.sales, s, t - k)
    return pd.DataFrame(out)


def _direct_frame(
    m: _Matrices,
    first: np.ndarray,
    cal: pd.DataFrame,
    s: np.ndarray,
    o: np.ndarray,
    h: np.ndarray,
) -> pd.DataFrame:
    """Features for origin columns ``o`` and horizon steps ``h`` (target o + h)."""
    t = o + h
    out = _common_features(m, first, cal, s, o, t)
    out["h"] = h.astype(float)
    out["last_1"] = _take(m.sales, s, o)
    # Most recent day on or before the origin with the target's weekday
    same_weekday = t - 7 * np.ceil(h / 7).astype(int)
    weeks = np.stack([_take(m.sales, s, same_weekday - 7 * k) for k in range(4)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # All-NaN weeks give NaN
        out["wd_mean_4w"] = np.nanmean(weeks, axis=0)
    out["lag_sw"] = weeks[0]
    for k in DIRECT_TARGET_LAGS:
        # Known at the origin only when k >= h; feature_columns keeps it only then
        out[f"lag_{k}"] = np.where(k >= h, _take(m.sales, s, t - k), np.nan)
    if (h > 0).all() and len(o):
        # Origin-anchored inputs never reach past the origin
        assert (same_weekday <= o).all()
    return pd.DataFrame(out)


def _with_category(frame: pd.DataFrame, n_series: int) -> pd.DataFrame:
    frame["series"] = pd.Categorical(frame["series"], categories=range(n_series))
    return frame


# Forecaster ----------------------------------------------------------------------------


class LightGBMFold:
    """
    Build features once per cutoff and fit/forecast any number of variants.

    Training frames are cached per (strategy, stockout handling); variants that differ
    only in feature subset or Tweedie power reuse them.
    """

    def __init__(self, orders: pd.DataFrame, cutoff: pd.Timestamp, horizon: int):
        self.cutoff = pd.Timestamp(cutoff).normalize()
        self.horizon = horizon
        self.orders = orders[orders["date"] <= self.cutoff]
        self.panel = build_panel(orders, self.cutoff)
        n_days = len(self.panel.dates)
        extended = pd.date_range(
            self.panel.dates[0], periods=n_days + horizon, freq="D"
        )
        self.calendar = calendar_features(extended)
        self.stockouts = flag_stockouts(self.panel.sales)
        self._matrices: Dict[str, _Matrices] = {}
        self._training: Dict[tuple, object] = {}
        self._validation: Dict[tuple, tuple] = {}
        self._direct_predictions: Dict[tuple, np.ndarray] = {}
        self.calibration_factors: Dict[tuple, float] = {}
        self.selected_power: Optional[float] = None

    # -- inputs

    def _feature_sales(self, stockout: str) -> np.ndarray:
        sales = self.panel.sales.copy()
        if stockout == "mask":
            sales[self.stockouts] = np.nan
        elif stockout != "none":
            raise ValueError(f"Unknown stockout handling: {stockout}")
        return sales

    def matrices(self, stockout: str) -> _Matrices:
        if stockout not in self._matrices:
            self._matrices[stockout] = _Matrices.from_grids(
                self._feature_sales(stockout), self.panel.price
            )
        return self._matrices[stockout]

    def _target_ok(self, stockout: str) -> np.ndarray:
        """Columns usable as training targets."""
        ok = ~np.isnan(self.panel.sales)
        if stockout == "mask":
            ok &= ~self.stockouts
        return ok

    # -- training data

    def _recursive_training(self, config: LGBMConfig):
        key = ("recursive", config.stockout, config.train_days)
        if key not in self._training:
            m = self.matrices(config.stockout)
            n_days = len(self.panel.dates)
            start = max(1, n_days - config.train_days)
            s, t = np.nonzero(self._target_ok(config.stockout)[:, start:])
            t = t + start
            frame = _recursive_frame(m, self.panel.first, self.calendar, s, t)
            # Rows of series with no sales in the prior 91 days carry no signal
            keep = frame["rmean_91"].to_numpy() > 0
            assert (t <= n_days - 1).all(), "Training target after the cutoff"
            frame = _with_category(
                frame[keep].reset_index(drop=True), len(self.panel.mlbs)
            )
            y = self.panel.sales[s[keep], t[keep]]
            self._training[key] = (frame, y)
        return self._training[key]

    def _direct_training(self, config: LGBMConfig, block: Tuple[int, int], step: int):
        key = ("direct", config.stockout, config.train_days, block, step)
        if key not in self._training:
            m = self.matrices(config.stockout)
            n_days = len(self.panel.dates)
            last_target = n_days - 1
            origins = np.arange(
                last_target - block[0], last_target - config.train_days, -step
            )
            origins = origins[origins >= 0]
            hs = np.arange(block[0], block[1] + 1)
            n_series = len(self.panel.mlbs)
            s, o, h = (
                a.ravel()
                for a in np.meshgrid(np.arange(n_series), origins, hs, indexing="ij")
            )
            t = o + h
            ok = (t <= last_target) & (o >= self.panel.first[s])
            s, o, h, t = s[ok], o[ok], h[ok], t[ok]
            ok = self._target_ok(config.stockout)[s, t]
            s, o, h, t = s[ok], o[ok], h[ok], t[ok]
            frame = _direct_frame(m, self.panel.first, self.calendar, s, o, h)
            keep = frame["rmean_91"].to_numpy() > 0
            assert (t[keep] <= last_target).all(), "Training target after the cutoff"
            assert (o[keep] < t[keep]).all()
            frame = _with_category(frame[keep].reset_index(drop=True), n_series)
            y = self.panel.sales[s[keep], t[keep]]
            self._training[key] = (frame, y)
        return self._training[key]

    # -- fitting

    def _fit(self, config: LGBMConfig, frame: pd.DataFrame, y: np.ndarray, columns):
        params = {
            "objective": "tweedie",
            "tweedie_variance_power": config.tweedie_power,
            "n_estimators": config.n_estimators,
            "learning_rate": config.learning_rate,
            "num_leaves": config.num_leaves,
            "min_child_samples": config.min_child_samples,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.8,
            "random_state": config.seed,
            "deterministic": True,
            "force_row_wise": True,
            "n_jobs": os.cpu_count(),
            "verbose": -1,
            **config.params,
        }
        model = lgb.LGBMRegressor(**params)
        categorical = [c for c in CATEGORICAL if c in columns]
        weight = None
        if config.target_scale:
            scale = np.clip(
                np.nan_to_num(
                    frame[f"rmean_{config.target_scale}"].to_numpy(), nan=0.0
                ),
                0.05,
                None,
            )
            y = y / scale
            weight = scale
        if config.weight_power:
            level = np.nan_to_num(frame["rmean_28"].to_numpy(), nan=0.0)
            extra = np.clip(level, 0.05, None) ** config.weight_power
            weight = extra if weight is None else weight * extra
        model.fit(
            frame[columns],
            y,
            sample_weight=weight,
            categorical_feature=categorical or "auto",
        )
        return model

    # -- forecasting

    def forecast(self, config: LGBMConfig, mlbs: List[str]) -> Dict[str, pd.Series]:
        """Forecasts for ``mlbs`` (series without sales by the cutoff are skipped)."""
        if config.strategy != "direct" and (
            config.calibration != "none" or config.target_scale
        ):
            raise ValueError(
                "Calibration and target scaling require the direct strategy"
            )
        if config.target_scale not in (0, 28, 91):
            raise ValueError("target_scale must be 0, 28, or 91")
        if (
            not 0 <= config.calibration_shrink <= 1
            or not 0.8 <= config.calibration_floor <= 1.5
        ):
            raise ValueError("Invalid calibration shrinkage or floor")
        index = {mlb: i for i, mlb in enumerate(self.panel.mlbs)}
        rows = np.array([index[mlb] for mlb in mlbs if mlb in index], dtype=int)
        if config.strategy == "recursive":
            values = self._forecast_recursive(config, rows)
        elif config.strategy == "direct":
            if config.calibration == "none":
                values = self._forecast_direct(config, rows)
            else:
                values = self._forecast_calibrated(config, rows)
        else:
            raise ValueError(f"Unknown strategy: {config.strategy}")
        dates = horizon_dates(self.cutoff, self.horizon)
        return {
            self.panel.mlbs[r]: pd.Series(values[i], index=dates)
            for i, r in enumerate(rows)
        }

    def _forecast_recursive(self, config: LGBMConfig, rows: np.ndarray) -> np.ndarray:
        frame, y = self._recursive_training(config)
        columns = feature_columns(config)
        model = self._fit(config, frame, y, columns)

        n_days = len(self.panel.dates)
        pad = np.full((len(self.panel.mlbs), self.horizon), np.nan)
        sales = np.hstack([self.matrices(config.stockout).sales, pad])
        # Assume the last known unit price holds over the horizon
        price = np.hstack(
            [self.panel.price, np.repeat(self.panel.price[:, -1:], self.horizon, 1)]
        )
        m = _Matrices.from_grids(sales, price)
        out = np.zeros((len(rows), self.horizon))
        for step in range(self.horizon):
            t = n_days + step
            feats = _recursive_frame(
                m, self.panel.first, self.calendar, rows, np.full(len(rows), t)
            )
            feats = _with_category(feats, len(self.panel.mlbs))
            pred = np.clip(model.predict(feats[columns]), 0, None)
            out[:, step] = pred
            # Feed the prediction back in and refresh the rolling means at column t
            sales[rows, t] = pred
            for window, matrix in ((7, m.rmean_7), (28, m.rmean_28), (91, m.rmean_91)):
                recent = sales[rows, t - window + 1 : t + 1]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    matrix[rows, t] = np.nanmean(recent, axis=1)
        return out

    def _forecast_direct(self, config: LGBMConfig, rows: np.ndarray) -> np.ndarray:
        key = (
            (
                replace(
                    config,
                    calibration="none",
                    calibration_shrink=1.0,
                    calibration_floor=0.8,
                ),
                tuple(rows),
            )
            if not config.params
            else None
        )
        if key is not None and key in self._direct_predictions:
            return self._direct_predictions[key].copy()
        m = self.matrices(config.stockout)
        origin = len(self.panel.dates) - 1
        out = np.zeros((len(rows), self.horizon))
        for block, step in zip(config.blocks, config.origin_steps):
            if block[0] > self.horizon:
                break
            frame, y = self._direct_training(config, block, step)
            columns = feature_columns(config, block)
            model = self._fit(config, frame, y, columns)
            hs = np.arange(block[0], min(block[1], self.horizon) + 1)
            s, h = (a.ravel() for a in np.meshgrid(rows, hs, indexing="ij"))
            feats = _direct_frame(
                m, self.panel.first, self.calendar, s, np.full(len(s), origin), h
            )
            feats = _with_category(feats, len(self.panel.mlbs))
            pred = np.clip(model.predict(feats[columns]), 0, None)
            if config.target_scale:
                scale = np.clip(
                    np.nan_to_num(
                        feats[f"rmean_{config.target_scale}"].to_numpy(), nan=0.0
                    ),
                    0.05,
                    None,
                )
                pred *= scale
            out[:, hs - 1] = pred.reshape(len(rows), len(hs))
        if key is not None:
            self._direct_predictions[key] = out.copy()
        return out

    def _validation_forecast(self, config: LGBMConfig, rows: np.ndarray):
        """Fit at c - horizon and score only the observed window ending at c."""
        inner_cutoff = self.cutoff - pd.Timedelta(days=self.horizon)
        key = (
            (
                replace(config, calibration_shrink=1.0, calibration_floor=0.8),
                inner_cutoff,
                tuple(rows),
            )
            if not config.params
            else None
        )
        if key is not None and key in self._validation:
            return self._validation[key]
        inner = LightGBMFold(self.orders, inner_cutoff, self.horizon)
        names = self.panel.mlbs[rows]
        inner_index = {name: i for i, name in enumerate(inner.panel.mlbs)}
        valid = np.array([i for i, name in enumerate(names) if name in inner_index])
        inner_rows = np.array([inner_index[names[i]] for i in valid], dtype=int)
        if not len(valid):
            return (
                valid,
                np.empty((0, self.horizon)),
                np.empty((0, self.horizon)),
                np.array([]),
            )
        pred = inner._forecast_direct(replace(config, calibration="none"), inner_rows)
        dates = horizon_dates(inner_cutoff, self.horizon)
        actual = self.panel.sales[rows[valid]][:, self.panel.dates.get_indexer(dates)]
        actual = np.nan_to_num(actual, nan=0.0)
        level = inner.matrices(config.stockout).rmean_28[inner_rows, -1]
        result = (valid, pred, actual, level)
        if key is not None:
            self._validation[key] = result
        return result

    @staticmethod
    def _tiers(level: np.ndarray) -> np.ndarray:
        """Top 20% by origin level, matching the harness's tier share."""
        top = np.zeros(len(level), dtype=bool)
        if len(level):
            top[
                np.argsort(-np.nan_to_num(level))[: max(1, round(len(level) * 0.2))]
            ] = True
        return top

    def _forecast_calibrated(self, config: LGBMConfig, rows: np.ndarray) -> np.ndarray:
        if config.calibration not in {
            "overall",
            "block",
            "tier",
            "block_tier",
            "power",
        }:
            raise ValueError(f"Unknown calibration: {config.calibration}")
        chosen = config
        if config.calibration == "power":
            scored = []
            for power in (1.1, 1.2, 1.3, 1.4, 1.5):
                candidate = replace(config, tweedie_power=power, calibration="none")
                _, pred, actual, _ = self._validation_forecast(candidate, rows)
                total = actual.sum()
                if total > 0:
                    bias = (pred.sum() - total) / total
                    wape = np.abs(pred - actual).sum() / total
                    scored.append((abs(bias) > 0.05, wape, abs(bias), power))
            if scored:
                eligible = [s for s in scored if not s[0]]
                # If no power meets the constraint, choose the least biased one.
                self.selected_power = (
                    min(eligible, key=lambda s: s[1])[3]
                    if eligible
                    else min(scored, key=lambda s: s[2])[3]
                )
                chosen = replace(config, tweedie_power=self.selected_power)
            return self._forecast_direct(chosen, rows)

        valid, pred, actual, level = self._validation_forecast(
            replace(config, calibration="none"), rows
        )
        result = self._forecast_direct(config, rows)
        self.calibration_factors = {}
        if not len(valid):
            return result
        use_block = config.calibration in {"block", "block_tier"}
        use_tier = config.calibration in {"tier", "block_tier"}
        validation_tier = self._tiers(level)
        current_level = self.matrices(config.stockout).rmean_28[rows, -1]
        current_tier = self._tiers(current_level)
        for block in config.blocks if use_block else ((1, self.horizon),):
            start, end = block[0] - 1, min(block[1], self.horizon)
            if start >= end:
                continue
            for tier in (False, True) if use_tier else (None,):
                mask = (
                    validation_tier == tier
                    if use_tier
                    else np.ones(len(valid), dtype=bool)
                )
                observed = actual[mask, start:end].sum()
                forecast = pred[mask, start:end].sum()
                # A minimum volume prevents a handful of intermittent days from
                # setting a noisy ratio. Shrink and bound the remaining ratios.
                raw_factor = (
                    np.clip(observed / forecast, 0.8, 1.5) if forecast >= 50 else 1.0
                )
                raw_factor = float(raw_factor) if np.isfinite(raw_factor) else 1.0
                factor = float(
                    np.clip(
                        max(
                            config.calibration_floor,
                            1 + config.calibration_shrink * (raw_factor - 1),
                        ),
                        0.8,
                        1.5,
                    )
                )
                self.calibration_factors[(block, tier)] = factor
                selected = (
                    current_tier == tier if use_tier else np.ones(len(rows), dtype=bool)
                )
                result[selected, start:end] *= factor
        return result


def forecast_lightgbm(
    orders: pd.DataFrame,
    cutoff: pd.Timestamp,
    horizon: int,
    mlbs: List[str],
    configs: Dict[str, LGBMConfig],
) -> Dict[str, Dict[str, pd.Series]]:
    """Forecasts per variant name for the given series at one cutoff."""
    fold = LightGBMFold(orders, cutoff, horizon)
    return {name: fold.forecast(config, mlbs) for name, config in configs.items()}


# Variants ------------------------------------------------------------------------------

UNCALIBRATED_CONFIG = LGBMConfig()
DEFAULT_CONFIG = replace(
    UNCALIBRATED_CONFIG,
    calibration="block_tier",
    calibration_shrink=0.35,
    calibration_floor=1.06,
)


def _variants() -> Dict[str, LGBMConfig]:
    """
    Named variants: the default direct and recursive models plus ablations.

    For each strategy: feature groups added in turn (``_fs_<groups>``), the default
    with one element removed or added (``_no_id``, ``_no_lag364``, ``_no_stockout``,
    ``_with_month``), and Tweedie variance powers 1.1-1.5 (``_p<power>``).
    """
    variants = {}
    for strategy in ("direct", "recursive"):
        base = replace(UNCALIBRATED_CONFIG, strategy=strategy)
        prefix = f"lgbm_{strategy}"
        variants[prefix] = base
        for n in range(1, len(FEATURE_GROUPS)):
            groups = FEATURE_GROUPS[:n]
            variants[f"{prefix}_fs_{'+'.join(groups)}"] = replace(
                base, feature_groups=groups
            )
        variants[f"{prefix}_no_id"] = replace(base, use_series_id=False)
        variants[f"{prefix}_no_lag364"] = replace(base, use_lag_364=False)
        variants[f"{prefix}_no_stockout"] = replace(base, stockout="none")
        variants[f"{prefix}_with_month"] = replace(base, drop_columns=())
        for power in (1.1, 1.2, 1.3, 1.4, 1.5):
            variants[f"{prefix}_p{power:.1f}"] = replace(base, tweedie_power=power)
    for method in ("overall", "block", "tier", "block_tier", "power"):
        variants[f"lgbm_direct_cal_{method}"] = replace(
            UNCALIBRATED_CONFIG, calibration=method
        )
    for window in (28, 91):
        variants[f"lgbm_direct_scaled_{window}"] = replace(
            UNCALIBRATED_CONFIG, target_scale=window
        )
    variants["lgbm_direct_uncalibrated"] = UNCALIBRATED_CONFIG
    variants["lgbm_direct"] = DEFAULT_CONFIG
    return variants


LGBM_VARIANTS: Dict[str, LGBMConfig] = _variants()
