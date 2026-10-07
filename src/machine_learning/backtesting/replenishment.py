"""Leakage-free protection-window forecasts and single-period ordering scores.

Dispersion uses method of moments on three weekly inner origins. Direct window
quantiles use weekly training origins in the year before each outer cutoff.
"""

import logging
import os
import time
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson

from backtesting.baselines import BASELINES
from backtesting.current_model import forecast_current_model
from backtesting.data_prep import build_training_features, eligible_series
from backtesting.lightgbm_model import (
    LGBM_VARIANTS,
    LightGBMFold,
    _direct_frame,
    _with_category,
)

logger = logging.getLogger(__name__)
ALPHAS = (0.5, 0.8, 0.9, 0.95)
MIN_TIER_PAIRS = 150
INNER_ORIGINS = 3
INNER_STEP_DAYS = 7
RATIO_FLOOR = 1.0
TIER_SHARE = 0.2
METHODS = ("nb_tier", "ratio_tier", "lgbm_quantile")


def inner_origins(cutoff, max_window, count=INNER_ORIGINS):
    """Origins whose largest protection window has ended by the outer cutoff."""
    cutoff = pd.Timestamp(cutoff)
    origins = [
        cutoff - pd.Timedelta(days=max_window + INNER_STEP_DAYS * k)
        for k in range(count)
    ]
    assert all(origin + pd.Timedelta(days=max_window) <= cutoff for origin in origins)
    return origins


def window_totals(daily, windows, actuals, truth=None):
    """Sum daily means and observed outcomes; daily quantiles are never inputs."""
    rows = []
    for (model, cutoff, mlb), frame in daily.groupby(["model", "cutoff", "mlb"]):
        frame = frame.sort_values("h")
        for window in windows:
            part = frame[frame.h <= window]
            if len(part) != window or part.forecast.isna().any():
                raise ValueError(
                    f"Missing daily forecast: {model}, {cutoff}, {mlb}, {window}"
                )
            row = {
                "model": model,
                "cutoff": cutoff,
                "mlb": mlb,
                "L": window,
                "tier": part.tier.iloc[0],
                "mu_W": float(part.forecast.sum()),
                "actual": float(part.actual.sum()),
                "in_common_set": bool(part.in_common_set.iloc[0]),
                "trailing_28": float(actuals[mlb].loc[:cutoff].tail(28).mean()),
            }
            if truth is not None:
                row["true_demand"] = float(part.true_demand.sum())
            rows.append(row)
    return pd.DataFrame(rows)


def _inner_window_rows(
    orders, actuals, cutoff, windows, models, eligibility_horizon, include_current=False
):
    """Fit each selected mean model once per inner origin for all window lengths."""
    rows = []
    longest = max(windows)
    for origin in inner_origins(cutoff, longest):
        features = build_training_features(orders, origin)
        eligible = eligible_series(features, origin, eligibility_horizon)
        if not eligible:
            continue
        histories = {mlb: actuals[mlb].loc[:origin] for mlb in eligible}
        volume = pd.Series({mlb: series.sum() for mlb, series in histories.items()})
        top = set(volume.nlargest(max(1, round(len(volume) * TIER_SHARE))).index)
        forecasts = {}
        for model in models:
            if model in BASELINES:
                forecasts[model] = {
                    mlb: BASELINES[model](histories[mlb], origin, longest)
                    for mlb in eligible
                }
        lgbm_models = [model for model in models if model in LGBM_VARIANTS]
        if lgbm_models:
            fold = LightGBMFold(orders, origin, longest)
            for model in lgbm_models:
                forecasts[model] = fold.forecast(LGBM_VARIANTS[model], eligible)
        if include_current and "current_xgboost" in models:
            forecasts["current_xgboost"] = forecast_current_model(
                features, eligible, origin, longest
            )
        for model, by_mlb in forecasts.items():
            missing = set(eligible) - set(by_mlb)
            if missing:
                raise ValueError(
                    f"{model}: {len(missing)} eligible inner series missing"
                )
            for mlb in eligible:
                daily = by_mlb[mlb].to_numpy(dtype=float)
                actual = (
                    actuals[mlb].reindex(by_mlb[mlb].index, fill_value=0).to_numpy()
                )
                for window in windows:
                    rows.append(
                        {
                            "model": model,
                            "cutoff": cutoff,
                            "origin": origin,
                            "mlb": mlb,
                            "L": window,
                            "tier": "top 20%" if mlb in top else "rest",
                            "mu_W": float(daily[:window].sum()),
                            "actual": float(actual[:window].sum()),
                        }
                    )
    return pd.DataFrame(rows)


def fit_dispersion(inner, cutoff, alphas=ALPHAS, minimum=MIN_TIER_PAIRS):
    """Moment NB spread and empirical ratios, pooling tiers when sparse."""
    rows = []
    for (model, window), group in inner.groupby(["model", "L"]):
        for tier in ("top 20%", "rest"):
            own = group[group.tier == tier]
            pooled = len(own) < minimum
            fit = group if pooled else own
            if pooled:
                logger.info(
                    "Pooled tiers for %s, L=%s, %s at %s: %s < %s pairs",
                    model,
                    window,
                    tier,
                    pd.Timestamp(cutoff).date(),
                    len(own),
                    minimum,
                )
            mu = fit.mu_W.to_numpy(dtype=float)
            actual = fit.actual.to_numpy(dtype=float)
            denominator = np.square(mu).sum()
            phi = (
                max(0.0, float((np.square(actual - mu) - mu).sum() / denominator))
                if denominator > 0
                else 0.0
            )
            ratios = actual[mu >= RATIO_FLOOR] / mu[mu >= RATIO_FLOOR]
            row = {
                "model": model,
                "cutoff": cutoff,
                "L": window,
                "tier": tier,
                "phi": phi,
                "n_pairs": len(own),
                "n_fit_pairs": len(fit),
                "n_ratios": len(ratios),
                "pooled_tiers": pooled,
            }
            for alpha in alphas:
                row[f"ratio_{alpha:g}"] = (
                    float(np.quantile(ratios, alpha)) if len(ratios) else np.nan
                )
            rows.append(row)
    return pd.DataFrame(rows)


def nb_quantile(mu, phi, alpha):
    """Integer NB quantile, with Poisson and zero-mean limits."""
    if mu <= 1e-10:
        return 0
    if phi <= 1e-10:
        return int(poisson.ppf(alpha, mu))
    size = 1.0 / phi
    return int(nbinom.ppf(alpha, size, size / (size + mu)))


def apply_spread(windows, dispersion, methods, alphas=ALPHAS):
    """Apply the same learned spread family to every selected mean forecast."""
    lookup = dispersion.set_index(["model", "cutoff", "L", "tier"])
    rows = []
    for row in windows.itertuples(index=False):
        fit = lookup.loc[(row.model, row.cutoff, row.L, row.tier)]
        base = row._asdict()
        for method in methods:
            if method == "lgbm_quantile":
                continue
            output = {**base, "method": method}
            for alpha in alphas:
                nb = nb_quantile(row.mu_W, fit.phi, alpha)
                ratio = fit[f"ratio_{alpha:g}"]
                output[f"q_{alpha:g}"] = (
                    int(np.ceil(max(0, row.mu_W * ratio)))
                    if method == "ratio_tier"
                    and row.mu_W >= RATIO_FLOOR
                    and np.isfinite(ratio)
                    else nb
                )
            rows.append(output)
    return pd.DataFrame(rows)


def _window_features(fold, series, origins, window):
    """Origin-anchored direct features and known calendar counts for a window."""
    m = fold.matrices("mask")
    frame = _direct_frame(
        m,
        fold.panel.first,
        fold.calendar,
        series,
        origins,
        np.ones(len(series), dtype=int),
    )
    sales = np.nan_to_num(fold.panel.sales, nan=0.0)
    cumulative = np.pad(np.cumsum(sales, axis=1), ((0, 0), (1, 0)))
    start = origins + 1 - 364
    end = origins + window + 1 - 364
    frame["lag_364_window"] = np.where(
        start >= 0,
        cumulative[series, np.clip(end, 0, cumulative.shape[1] - 1)]
        - cumulative[series, np.clip(start, 0, cumulative.shape[1] - 1)],
        np.nan,
    )
    cal = fold.calendar
    for name in ("br_holiday", "bf_week", "bf_peak", "xmas_runup", "xmas_hol"):
        cumulative_cal = np.r_[0, cal[name].to_numpy().cumsum()]
        frame[f"{name}_count"] = (
            cumulative_cal[origins + window + 1] - cumulative_cal[origins + 1]
        )
    frame["window"] = window
    return _with_category(frame, len(fold.panel.mlbs))


def direct_window_quantiles(orders, cutoff, windows, alphas, mlbs, fast=None):
    """Fit one quantile LightGBM per (window, alpha), using past window targets."""
    fold = LightGBMFold(orders, cutoff, max(windows))
    n_days = len(fold.panel.dates)
    index = {mlb: i for i, mlb in enumerate(fold.panel.mlbs)}
    requested = [mlb for mlb in mlbs if mlb in index]
    if len(requested) != len(mlbs):
        raise ValueError("Direct quantile model is missing eligible series")
    rows = np.array([index[mlb] for mlb in requested])
    sales = np.nan_to_num(fold.panel.sales, nan=0.0)
    cumulative = np.pad(np.cumsum(sales, axis=1), ((0, 0), (1, 0)))
    output = []
    settings = {"n_estimators": 120, "num_leaves": 31, "min_child_samples": 50}
    if fast:
        settings.update(fast)
    for window in windows:
        last_origin = n_days - window - 1
        first_origin = max(0, n_days - 366)
        origins = np.arange(last_origin, first_origin - 1, -7)
        s, o = (
            array.ravel()
            for array in np.meshgrid(
                np.arange(len(fold.panel.mlbs)), origins, indexing="ij"
            )
        )
        keep = (o >= fold.panel.first[s]) & (o + window < n_days)
        s, o = s[keep], o[keep]
        assert (o + window < n_days).all(), "Training window extends beyond cutoff"
        train = _window_features(fold, s, o, window)
        target = cumulative[s, o + window + 1] - cumulative[s, o + 1]
        keep = train.rmean_91.to_numpy() > 0
        train, target = train.loc[keep].reset_index(drop=True), target[keep]
        predict = _window_features(fold, rows, np.full(len(rows), n_days - 1), window)
        columns = [
            "last_1",
            "rmean_7",
            "rmean_28",
            "rmean_91",
            "wd_mean_4w",
            "price",
            "price_rel_28",
            "price_rel_max_90",
            "age",
            "series",
            "lag_364_window",
            "br_holiday_count",
            "bf_week_count",
            "bf_peak_count",
            "xmas_runup_count",
            "xmas_hol_count",
            "window",
        ]
        predictions = []
        for alpha in alphas:
            model = lgb.LGBMRegressor(
                objective="quantile",
                alpha=alpha,
                learning_rate=0.05,
                random_state=42,
                deterministic=True,
                force_row_wise=True,
                n_jobs=min(8, os.cpu_count() or 1),
                verbose=-1,
                **settings,
            )
            model.fit(train[columns], target, categorical_feature=["series"])
            predictions.append(np.clip(model.predict(predict[columns]), 0, None))
        ordered = np.ceil(np.maximum.accumulate(np.stack(predictions), axis=0))
        for i, mlb in enumerate(requested):
            row = {"mlb": mlb, "L": window}
            row.update(
                {f"q_{alpha:g}": int(ordered[j, i]) for j, alpha in enumerate(alphas)}
            )
            output.append(row)
    return pd.DataFrame(output)


def score_replenishment(forecasts, alphas=ALPHAS):
    """Window accuracy, coverage, pinball, and one-period order-up-to metrics."""
    rows = []
    targets = ["observed"] + (["true_demand"] if "true_demand" in forecasts else [])
    for target in targets:
        column = "actual" if target == "observed" else target
        for (model, method, window), group in forecasts.groupby(
            ["model", "method", "L"]
        ):
            subsets = [("overall", "all", group)]
            subsets += [("tier", name, part) for name, part in group.groupby("tier")]
            subsets += [
                ("cutoff", str(pd.Timestamp(cutoff).date()), part)
                for cutoff, part in group.groupby("cutoff")
            ]
            for scope, label, part in subsets:
                actual = part[column].to_numpy(dtype=float)
                mu = part.mu_W.to_numpy(dtype=float)
                total = actual.sum()
                for alpha in alphas:
                    q = part[f"q_{alpha:g}"].to_numpy(dtype=float)
                    shortfall = np.maximum(actual - q, 0)
                    excess = np.maximum(q - actual, 0)
                    fallback = part.trailing_28.to_numpy(dtype=float)
                    daily = np.where(actual > 0, actual / window, fallback)
                    days = np.divide(
                        excess, daily, out=np.zeros_like(excess), where=daily > 0
                    )
                    rows.append(
                        {
                            "model": model,
                            "method": method,
                            "L": window,
                            "alpha": alpha,
                            "target": target,
                            "scope": scope,
                            "group": label,
                            "n": len(part),
                            "mean_wape": np.abs(mu - actual).sum() / total
                            if total
                            else np.nan,
                            "mean_bias": (mu - actual).sum() / total
                            if total
                            else np.nan,
                            "coverage": np.mean(actual <= q),
                            "coverage_gap": np.mean(actual <= q) - alpha,
                            "scaled_pinball": (
                                np.maximum(
                                    alpha * (actual - q), (alpha - 1) * (actual - q)
                                ).sum()
                                / total
                                if total
                                else np.nan
                            ),
                            "fill_rate": 1 - shortfall.sum() / total
                            if total
                            else np.nan,
                            "stockout_rate": np.mean(shortfall > 0),
                            "excess_units": excess.sum() / total if total else np.nan,
                            "days_cover_held": float(np.mean(days)),
                        }
                    )
    scores = pd.DataFrame(rows)
    groups = ["model", "method", "L", "target", "scope", "group"]
    scores["mean_scaled_pinball"] = scores.groupby(groups)["scaled_pinball"].transform(
        "mean"
    )
    return scores


@dataclass
class ReplenishmentResult:
    forecasts: pd.DataFrame
    dispersion: pd.DataFrame
    scores: pd.DataFrame
    runtime_seconds: float


def run_replenishment(
    orders,
    daily,
    actuals,
    windows,
    alphas,
    methods,
    include_current=False,
    quantile_fast=None,
):
    """Add leakage-free window quantiles to an existing daily backtest."""
    start = time.monotonic()
    models = list(daily.model.unique())
    inner_models = [m for m in models if include_current or m != "current_xgboost"]
    eligible_daily = daily[daily.model.isin(inner_models)]
    for cutoff, group in eligible_daily[eligible_daily.in_common_set].groupby("cutoff"):
        expected = set(group.mlb.unique())
        for model in inner_models:
            present = set(group[group.model == model].mlb.unique())
            if present != expected:
                raise ValueError(
                    f"{model} at {cutoff}: {len(expected - present)} common series missing"
                )
    outer = window_totals(eligible_daily, windows, actuals)
    if "true_demand" in daily:
        truth_totals = window_totals(eligible_daily, windows, actuals, truth=True)
        outer["true_demand"] = truth_totals.true_demand
    outer = outer[outer.in_common_set].reset_index(drop=True)
    dispersions = []
    for cutoff in sorted(outer.cutoff.unique()):
        inner = _inner_window_rows(
            orders,
            actuals,
            cutoff,
            windows,
            inner_models,
            int(daily.h.max()),
            include_current,
        )
        dispersions.append(fit_dispersion(inner, cutoff, alphas))
    dispersion = pd.concat(dispersions, ignore_index=True)
    forecast = apply_spread(outer, dispersion, methods, alphas)
    if "lgbm_quantile" in methods and any(m in LGBM_VARIANTS for m in models):
        challenger = []
        for cutoff, group in outer.groupby("cutoff"):
            names = sorted(group.mlb.unique())
            quantiles = direct_window_quantiles(
                orders, cutoff, windows, alphas, names, fast=quantile_fast
            )
            template_name = (
                "lgbm_direct"
                if "lgbm_direct" in models
                else next(m for m in models if m in LGBM_VARIANTS)
            )
            template = group[group.model == template_name].drop(columns=["model"])
            merged = template.merge(quantiles, on=["mlb", "L"], validate="one_to_one")
            if len(merged) != len(template):
                raise ValueError("Missing direct quantile window forecast")
            challenger.append(
                merged.assign(model="lgbm_quantile", method="lgbm_quantile")
            )
        forecast = pd.concat([forecast, *challenger], ignore_index=True)
    expected = outer.groupby(["cutoff", "L", "model"]).size()
    actual_counts = forecast.groupby(["cutoff", "L", "model", "method"]).size()
    for key, count in actual_counts.items():
        if key[2] != "lgbm_quantile" and count != expected.loc[key[:3]]:
            raise ValueError(f"Missing window forecast for {key}")
    scores = score_replenishment(forecast, alphas)
    return ReplenishmentResult(forecast, dispersion, scores, time.monotonic() - start)
