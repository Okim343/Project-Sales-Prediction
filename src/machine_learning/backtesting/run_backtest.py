"""
Rolling-origin backtest: the production model and challengers against simple baselines.

For each cutoff, every model is trained only on data up to the cutoff and scored on the
following ``horizon`` days. Series eligibility follows the production rule (sales within
``AppConfig.ACTIVE_MLB_DAYS_THRESHOLD`` days of the cutoff and at least horizon + 15
rows of history).

Two leaderboards are produced:
- **common set**: the top ``--max-series`` eligible series by training volume, scored
  for every model including the production model;
- **all eligible series**: every model except the production model (which is only run
  on the common set to save time).

With ``--source vendas`` the Greycom invoice sales (NF-e series 2) are loaded at SKU
level without the World Cup items, truncated at ``--end-date`` before cutoffs are
computed, and also scored by value-weighted WAPE (each series' last known unit price at
the cutoff as weight) and by cutoff.

With ``--true-demand`` the synthetic source is additionally scored against the planted
true demand (before stockouts censor it) from ``data/synthetic_truth.parquet``.

Usage (from src/machine_learning):
    python backtesting/run_backtest.py --source both
    python backtesting/run_backtest.py --source real --skip-current
    python backtesting/run_backtest.py --source synthetic --max-series 30 --folds 3
    python backtesting/run_backtest.py --models "baselines,lgbm_direct_p*" --true-demand
    python backtesting/run_backtest.py --source vendas --end-date 2025-12-31 --tag A

Outputs go to bld/backtest/<source>[_<tag>]_<timestamp>/: forecasts.parquet, scores.csv,
summary.md, and calibration_factors.csv when relevant.
"""

import argparse
import fnmatch
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import AppConfig  # noqa: E402
from backtesting.baselines import BASELINES  # noqa: E402
from backtesting.combinations import COMBINATIONS, add_combinations  # noqa: E402
from backtesting.current_model import forecast_current_model  # noqa: E402
from backtesting.data_prep import (  # noqa: E402
    build_actuals_grid,
    build_training_features,
    eligible_series,
    last_unit_price,
    load_orders,
)
from backtesting.folds import horizon_dates, make_cutoffs  # noqa: E402
from backtesting.lightgbm_model import LGBM_VARIANTS, LightGBMFold  # noqa: E402
from backtesting.metrics import seasonal_naive_scale, summarise  # noqa: E402
from backtesting.replenishment import (  # noqa: E402
    METHODS,
    run_replenishment,
)
from backtesting.synthetic_data import (  # noqa: E402
    DATA_DIR,
    PROJECT_ROOT,
    generate_synthetic_orders,
    save_synthetic,
)
from backtesting.vendas_data import (  # noqa: E402
    VENDAS_CSV,
    VENDAS_END_DATE,
    load_vendas_orders,
)

CURRENT_MODEL_NAME = "current_xgboost"
TOP_TIER_SHARE = 0.2
DEFAULT_MODELS = "baselines,current_xgboost,lgbm_direct,lgbm_recursive"
CONCENTRATION_TOP = 5
ALL_MODELS = [*BASELINES, CURRENT_MODEL_NAME, *LGBM_VARIANTS, *COMBINATIONS]

logger = logging.getLogger(__name__)


def resolve_models(spec: str) -> List[str]:
    """
    Model names from a comma-separated list of names or glob patterns.

    ``baselines`` expands to every baseline; patterns such as ``lgbm_direct_p*`` match
    any registered model.
    """
    names: List[str] = []
    for token in (t.strip() for t in spec.split(",") if t.strip()):
        pattern = "|".join(BASELINES) if token == "baselines" else token
        matches = [
            m
            for m in ALL_MODELS
            if any(fnmatch.fnmatchcase(m, p) for p in pattern.split("|"))
        ]
        if not matches:
            raise ValueError(f"Unknown model: {token}. Known: {', '.join(ALL_MODELS)}")
        names += [m for m in matches if m not in names]
    return names


def run_fold(
    orders: pd.DataFrame,
    actuals_by_mlb: Dict[str, pd.Series],
    cutoff: pd.Timestamp,
    horizon: int,
    max_series: int,
    models: List[str],
    truth_by_mlb: Optional[Dict[str, pd.Series]] = None,
    calibration_log: Optional[List[dict]] = None,
    price_weights: bool = False,
) -> pd.DataFrame:
    """
    Forecast and collect actuals for one cutoff; returns long-format rows.

    With ``price_weights`` each row also carries ``price_weight``, the series' last
    known unit price on or before the cutoff, for value-weighted scores.
    """
    features = build_training_features(orders, cutoff)
    eligible = eligible_series(features, cutoff, horizon)

    histories = {mlb: actuals_by_mlb[mlb][:cutoff] for mlb in eligible}
    volume = pd.Series({mlb: h.sum() for mlb, h in histories.items()}).sort_values(
        ascending=False
    )
    common = set(volume.index[:max_series])
    n_top = max(1, int(round(len(volume) * TOP_TIER_SHARE)))
    top_tier = set(volume.index[:n_top])
    dates = horizon_dates(cutoff, horizon)
    prices = last_unit_price(orders, cutoff) if price_weights else None
    combined = {m: COMBINATIONS[m] for m in models if m in COMBINATIONS}
    needed = list(
        dict.fromkeys(models + [n for spec in combined.values() for n in spec.inputs])
    )

    forecasts: Dict[str, Dict[str, pd.Series]] = {
        name: {mlb: fn(histories[mlb], cutoff, horizon) for mlb in eligible}
        for name, fn in BASELINES.items()
        if name in needed
    }
    lgbm_names = [m for m in needed if m in LGBM_VARIANTS]
    if lgbm_names:
        fold = LightGBMFold(orders, cutoff, horizon)
        for name in lgbm_names:
            start = time.time()
            forecasts[name] = fold.forecast(LGBM_VARIANTS[name], eligible)
            logger.info(f"{name}: {len(eligible)} series in {time.time() - start:.0f}s")
            for item in fold.calibration_diagnostics:
                record = {
                    "cutoff": cutoff,
                    "inner_cutoff": (
                        cutoff - pd.Timedelta(days=horizon)
                        if LGBM_VARIANTS[name].calibration != "constant"
                        else pd.NaT
                    ),
                    "model": name,
                    **item,
                }
                if calibration_log is not None:
                    calibration_log.append(record)
                logger.info("Calibration: %s", record)
    if CURRENT_MODEL_NAME in needed:
        start = time.time()
        forecasts[CURRENT_MODEL_NAME] = forecast_current_model(
            features, sorted(common), cutoff, horizon
        )
        logger.info(
            f"Production model: {len(common)} series in {time.time() - start:.0f}s"
        )

    forecasts = {
        name: by_mlb
        for name, by_mlb in add_combinations(forecasts, combined).items()
        if name in models
    }

    rows = []
    for mlb in eligible:
        actual = actuals_by_mlb[mlb].reindex(dates, fill_value=0).to_numpy()
        true_demand = (
            truth_by_mlb[mlb].reindex(dates, fill_value=0).to_numpy()
            if truth_by_mlb is not None
            else None
        )
        scale = seasonal_naive_scale(histories[mlb])
        for model, by_mlb in forecasts.items():
            if mlb not in by_mlb:
                continue
            rows.append(
                pd.DataFrame(
                    {
                        "model": model,
                        "cutoff": cutoff,
                        "mlb": mlb,
                        "date": dates,
                        "h": range(1, horizon + 1),
                        "forecast": by_mlb[mlb].reindex(dates).to_numpy(),
                        "actual": actual,
                        "scale": scale,
                        "tier": "top 20%" if mlb in top_tier else "rest",
                        "in_common_set": mlb in common,
                        **({"true_demand": true_demand} if truth_by_mlb else {}),
                        **({"price_weight": prices[mlb]} if prices is not None else {}),
                    }
                )
            )
    logger.info(
        f"Cutoff {cutoff.date()}: {len(eligible)} eligible series, "
        f"{len(common)} in common set"
    )
    return pd.concat(rows, ignore_index=True)


def truncate_and_cut(
    orders: pd.DataFrame, args: argparse.Namespace
) -> Tuple[pd.DataFrame, List[pd.Timestamp]]:
    """
    Orders up to ``args.end_date`` (when set) and the fold cutoffs computed after that.

    With an end date the last cutoff is end date - horizon, even if the last day has no
    sales; without one it is the last order date - horizon.
    """
    end_date = getattr(args, "end_date", None)
    if end_date is not None:
        end_date = pd.Timestamp(end_date).normalize()
        orders = orders[orders["date"] <= end_date]
    cutoffs = make_cutoffs(
        orders["date"].min(),
        end_date if end_date is not None else orders["date"].max(),
        args.horizon,
        args.folds,
        args.step_days,
    )
    return orders, cutoffs


def run_source(
    name: str,
    orders: pd.DataFrame,
    args: argparse.Namespace,
    truth: Optional[pd.DataFrame] = None,
    value_weights: bool = False,
) -> Path:
    """
    Backtest one data source and write its outputs.

    ``value_weights`` adds value-weighted WAPE, per-cutoff scores and a unit
    concentration table to the outputs.
    """
    models = resolve_models(args.models)
    if args.skip_current and CURRENT_MODEL_NAME in models:
        models.remove(CURRENT_MODEL_NAME)
    truth_by_mlb = (
        {
            mlb: frame.set_index("date")["true_demand"]
            for mlb, frame in truth.groupby("mlb")
        }
        if truth is not None
        else None
    )
    orders, cutoffs = truncate_and_cut(orders, args)
    actuals = build_actuals_grid(orders)
    actuals_by_mlb = {
        mlb: frame.set_index("date")["y"] for mlb, frame in actuals.groupby("mlb")
    }
    logger.info(f"[{name}] cutoffs: {[c.date().isoformat() for c in cutoffs]}")

    calibration_log: List[dict] = []
    results = pd.concat(
        [
            run_fold(
                orders,
                actuals_by_mlb,
                c,
                args.horizon,
                args.max_series,
                models,
                truth_by_mlb,
                calibration_log,
                price_weights=value_weights,
            )
            for c in cutoffs
        ],
        ignore_index=True,
    )

    common = results[results["in_common_set"]]
    every_series = results[results["model"] != CURRENT_MODEL_NAME]
    frames = [
        summarise(common, by_cutoff=value_weights).assign(
            scope=f"common set (top {args.max_series})"
        ),
        summarise(every_series, by_cutoff=value_weights).assign(
            scope="all eligible series"
        ),
    ]
    without = [
        m.strip()
        for m in (getattr(args, "score_without", None) or "").split(",")
        if m.strip()
    ]
    if without:
        frames.append(
            summarise(
                common[~common["mlb"].isin(without)], by_cutoff=value_weights
            ).assign(
                scope=f"common set (top {args.max_series}) without {', '.join(without)}"
            )
        )
    if truth_by_mlb is not None:
        frames.append(
            summarise(common.assign(actual=common["true_demand"])).assign(
                scope=f"common set (top {args.max_series}), scored on true demand"
            )
        )
    scores = pd.concat(frames, ignore_index=True)

    label = f"{name}_{args.tag}" if getattr(args, "tag", None) else name
    out_dir = (
        PROJECT_ROOT / "bld" / "backtest" / f"{label}_{datetime.now():%Y%m%d_%H%M%S}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_parquet(out_dir / "forecasts.parquet", index=False)
    scores.to_csv(out_dir / "scores.csv", index=False)
    if calibration_log:
        pd.DataFrame(calibration_log).to_csv(
            out_dir / "calibration_factors.csv", index=False
        )
    summary = render_summary(label, cutoffs, scores, args)
    if value_weights:
        concentration = pd.concat(
            [
                unit_concentration(common).assign(
                    scope=f"common set (top {args.max_series})"
                ),
                unit_concentration(every_series).assign(scope="all eligible series"),
            ],
            ignore_index=True,
        )
        concentration.to_csv(out_dir / "concentration.csv", index=False)
        summary += render_concentration(results, concentration)
    if getattr(args, "replenishment", False):
        windows = _numbers(args.windows, int)
        alphas = _numbers(args.service_levels, float)
        methods = [m.strip() for m in args.quantile_methods.split(",") if m.strip()]
        if not windows or min(windows) < 1 or max(windows) > args.horizon:
            raise ValueError("--windows must contain days between 1 and --horizon")
        if not alphas or not all(0 < alpha < 1 for alpha in alphas):
            raise ValueError("--service-levels must be between 0 and 1")
        if any(method not in METHODS for method in methods):
            raise ValueError(f"--quantile-methods must use {METHODS}")
        replenishment = run_replenishment(
            orders,
            results,
            actuals_by_mlb,
            windows,
            alphas,
            methods,
            include_current=args.replenishment_include_current,
        )
        replenishment.forecasts.to_parquet(
            out_dir / "window_forecasts.parquet", index=False
        )
        replenishment.dispersion.to_csv(out_dir / "dispersion.csv", index=False)
        replenishment.scores.to_csv(out_dir / "replenishment_scores.csv", index=False)
        summary += render_replenishment(replenishment, args)
    (out_dir / "summary.md").write_text(summary)
    print(summary)
    return out_dir


def render_summary(
    name: str,
    cutoffs: List[pd.Timestamp],
    scores: pd.DataFrame,
    args: argparse.Namespace,
) -> str:
    """Markdown leaderboard plus horizon-bucket and volume-tier tables."""
    lines = [
        f"# Backtest: {name}",
        "",
        f"Horizon {args.horizon} days; cutoffs "
        + ", ".join(c.date().isoformat() for c in cutoffs)
        + ". WAPE and MASE: lower is better. Bias > 0 means over-forecasting.",
        "",
    ]
    weighted = "wape_value" in scores
    if weighted:
        lines += [
            "wape_value weights each series' errors by its last known unit price on or "
            "before the cutoff (revenue-weighted WAPE).",
            "",
        ]
    metrics = ["wape", "wape_value"] if weighted else ["wape"]
    names = {"wape": "WAPE", "wape_value": "wape_value", "bias": "Bias"}
    groups = {"horizon": "horizon", "volume": "volume tier", "cutoff": "cutoff"}
    for scope, frame in scores.groupby("scope", sort=False):
        overall = frame[frame["group_type"] == "overall"].sort_values("wape")
        lines += [
            f"## Leaderboard: {scope}",
            "",
            _table(overall, ["model", *metrics, "mase", "bias", "series_x_folds"]),
            "",
        ]
        tables = [
            (group_type, metric)
            for group_type in groups
            if (frame["group_type"] == group_type).any()
            for metric in metrics + (["bias"] if group_type == "cutoff" else [])
        ]
        for group_type, metric in tables:
            title = f"{names[metric]} by {groups[group_type]}"
            pivot = (
                frame[frame["group_type"] == group_type]
                .pivot(index="model", columns="group", values=metric)
                .reindex(
                    index=overall["model"], columns=_ordered_groups(frame, group_type)
                )
            )
            lines += [
                f"### {title}",
                "",
                _table(pivot.reset_index(), ["model", *pivot.columns]),
                "",
            ]
    return "\n".join(lines)


def unit_concentration(
    results: pd.DataFrame, top: int = CONCENTRATION_TOP
) -> pd.DataFrame:
    """
    The ``top`` series by scored actual units, with their unit and value shares.

    Actuals are the same for every model, so one model's rows are used. Value uses
    the ``price_weight`` column.
    """
    rows = results[results["model"] == results["model"].iloc[0]]
    rows = rows.assign(value=rows["actual"] * rows["price_weight"])
    totals = rows.groupby("mlb")[["actual", "value"]].sum()
    shares = (totals / totals.sum()).sort_values("actual", ascending=False).head(top)
    return shares.rename(
        columns={"actual": "unit_share", "value": "value_share"}
    ).reset_index()


def render_concentration(results: pd.DataFrame, concentration: pd.DataFrame) -> str:
    """Markdown table of the top series' share of scored units."""
    lines = ["", "## Unit concentration", ""]
    for scope, frame in concentration.groupby("scope", sort=False):
        lines += [
            f"### Top {len(frame)} series by scored units: {scope}",
            "",
            _table(frame, ["mlb", "unit_share", "value_share"]),
            "",
            f"Together: {frame['unit_share'].sum():.3f} of units, "
            f"{frame['value_share'].sum():.3f} of value.",
            "",
        ]
    return "\n".join(lines)


def _numbers(spec, kind):
    return list(
        dict.fromkeys(kind(token.strip()) for token in spec.split(",") if token.strip())
    )


def render_replenishment(result, args):
    """Show the primary ordering trade-off without hiding the full CSV scores."""
    scores = result.scores
    frame = scores[
        (scores.method == "nb_tier")
        & (scores.target == "observed")
        & (scores.scope == "overall")
    ]
    columns = [
        "model",
        "L",
        "alpha",
        "mean_wape",
        "mean_bias",
        "coverage",
        "scaled_pinball",
        "fill_rate",
        "stockout_rate",
        "excess_units",
    ]
    lines = [
        "",
        "## Replenishment",
        "",
        f"Protection windows {args.windows} days and service levels {args.service_levels} "
        "are placeholders pending business inputs.",
        "Window NB spread is fitted by method of moments on three weekly inner origins; "
        "tiers with fewer than 150 pairs pool both tiers. Quantiles are integer units. "
        "The direct LightGBM window quantiles are rounded up to integer units.",
        "Real outcomes are observed sales and may be stockout-censored. The ordering "
        "simulation is single-period.",
        f"Replenishment runtime: {result.runtime_seconds / 60:.1f} minutes.",
    ]
    if (
        "current_xgboost" in resolve_models(args.models)
        and not args.replenishment_include_current
    ):
        lines.append(
            "current_xgboost was excluded from replenishment inner folds for runtime."
        )
    lines += [
        "",
        "### NB ordering trade-off (observed sales)",
        "",
        _table(frame, columns),
        "",
    ]
    return "\n".join(lines)


def _ordered_groups(frame: pd.DataFrame, group_type: str) -> List[str]:
    """Group labels in their natural order (as first produced by summarise)."""
    return list(dict.fromkeys(frame.loc[frame["group_type"] == group_type, "group"]))


def _table(frame: pd.DataFrame, columns: List[str]) -> str:
    """Render a DataFrame as a Markdown table with 3-decimal floats."""
    header = "| " + " | ".join(columns) + " |"
    divider = "|" + "|".join("---" for _ in columns) + "|"
    body = [
        "| "
        + " | ".join(
            f"{row[c]:.3f}" if isinstance(row[c], float) else str(row[c])
            for c in columns
        )
        + " |"
        for _, row in frame.iterrows()
    ]
    return "\n".join([header, divider, *body])


def parse_args(argv=None) -> argparse.Namespace:
    """Command-line options."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--source",
        choices=["real", "synthetic", "both", "db", "vendas"],
        default="both",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Order export; with --source db, use this instead of a live read",
    )
    parser.add_argument(
        "--synthetic-csv",
        type=Path,
        default=DATA_DIR / "synthetic_orders.csv",
        help="Synthetic orders; generated with the default seed if missing",
    )
    parser.add_argument(
        "--vendas-csv",
        type=Path,
        default=VENDAS_CSV,
        help="Greycom SKU-day invoice sales used by --source vendas",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="Last day of data; later orders are dropped before cutoffs are computed "
        f"(default for --source vendas: {VENDAS_END_DATE})",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Suffix for the output folder name, e.g. A gives vendas_A_<timestamp>",
    )
    parser.add_argument(
        "--score-without",
        default=None,
        help="Comma-separated series also left out of an extra common-set score, "
        "e.g. a SKU that dominates unit totals",
    )
    parser.add_argument(
        "--max-series",
        type=int,
        default=100,
        help="Series in the common set scored for the production model",
    )
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--step-days", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=AppConfig.FORECAST_DAYS_LONG)
    parser.add_argument(
        "--models",
        default=DEFAULT_MODELS,
        help="Comma-separated model names or glob patterns ('baselines' selects all "
        f"baselines). Default: {DEFAULT_MODELS}",
    )
    parser.add_argument(
        "--skip-current",
        action="store_true",
        help="Drop the production model (slowest) from --models",
    )
    parser.add_argument(
        "--true-demand",
        action="store_true",
        help="Also score the synthetic source against planted true demand",
    )
    parser.add_argument(
        "--truth",
        type=Path,
        default=DATA_DIR / "synthetic_truth.parquet",
        help="Synthetic ground truth used by --true-demand",
    )
    parser.add_argument("--replenishment", action="store_true")
    parser.add_argument("--windows", default="7,14,28")
    parser.add_argument("--service-levels", default="0.5,0.8,0.9,0.95")
    parser.add_argument(
        "--quantile-methods", default="nb_tier,ratio_tier,lgbm_quantile"
    )
    parser.add_argument("--replenishment-include-current", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    """Run the backtest for the requested sources."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logging.getLogger("data_management").setLevel(logging.WARNING)
    args = parse_args(argv)

    sources = {}
    if args.source in ("real", "both"):
        real_csv = args.csv or DATA_DIR / "raw_sql.csv"
        if not real_csv.exists():
            logger.error(f"Historical export not found: {real_csv}")
            return 1
        sources["real"] = load_orders(real_csv)
    if args.source == "db":
        if args.csv:
            sources["db"] = load_orders(args.csv, date_convention="production")
        else:
            from database_utils import db_manager

            sources["db"] = load_orders(
                db_manager.import_data_from_sql(), date_convention="production"
            )
    if args.source == "vendas":
        if args.end_date is None:
            args.end_date = VENDAS_END_DATE
        sources["vendas"] = load_vendas_orders(args.vendas_csv, end_date=args.end_date)
    if args.source in ("synthetic", "both"):
        if not args.synthetic_csv.exists():
            logger.info("Synthetic data not found; generating with the default seed")
            orders, truth = generate_synthetic_orders()
            save_synthetic(orders, truth, args.synthetic_csv.parent)
        sources["synthetic"] = load_orders(args.synthetic_csv)

    for name, orders in sources.items():
        truth = None
        if name == "synthetic" and args.true_demand:
            truth = pd.read_parquet(args.truth, columns=["mlb", "date", "true_demand"])
        out_dir = run_source(name, orders, args, truth, value_weights=name == "vendas")
        logger.info(f"[{name}] results written to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
