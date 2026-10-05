"""
Rolling-origin backtest: the production model against simple baselines.

For each cutoff, every model is trained only on data up to the cutoff and scored on the
following ``horizon`` days. Series eligibility follows the production rule (sales within
``AppConfig.ACTIVE_MLB_DAYS_THRESHOLD`` days of the cutoff and at least horizon + 15
rows of history).

Two leaderboards are produced:
- **common set**: the top ``--max-series`` eligible series by training volume, scored
  for every model including the production model;
- **all eligible series**: baselines only.

Usage (from src/machine_learning):
    python backtesting/run_backtest.py --source both
    python backtesting/run_backtest.py --source real --skip-current
    python backtesting/run_backtest.py --source synthetic --max-series 30 --folds 3

Outputs go to bld/backtest/<source>_<timestamp>/: forecasts.parquet, scores.csv, and
summary.md.
"""

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import AppConfig  # noqa: E402
from backtesting.baselines import BASELINES  # noqa: E402
from backtesting.current_model import forecast_current_model  # noqa: E402
from backtesting.data_prep import (  # noqa: E402
    build_actuals_grid,
    build_training_features,
    load_orders,
)
from backtesting.folds import horizon_dates, make_cutoffs  # noqa: E402
from backtesting.metrics import seasonal_naive_scale, summarise  # noqa: E402
from backtesting.synthetic_data import (  # noqa: E402
    DATA_DIR,
    PROJECT_ROOT,
    generate_synthetic_orders,
    save_synthetic,
)

CURRENT_MODEL_NAME = "current_xgboost"
TOP_TIER_SHARE = 0.2

logger = logging.getLogger(__name__)


def eligible_series(
    features: pd.DataFrame, cutoff: pd.Timestamp, horizon: int
) -> List[str]:
    """Series the production pipeline would forecast at this cutoff."""
    activity_cutoff = cutoff - pd.Timedelta(days=AppConfig.ACTIVE_MLB_DAYS_THRESHOLD)
    stats = features.groupby("mlb").apply(
        lambda frame: pd.Series({"last": frame.index.max(), "rows": len(frame)}),
        include_groups=False,
    )
    keep = (stats["last"] >= activity_cutoff) & (stats["rows"] >= horizon + 15)
    return stats.index[keep].tolist()


def run_fold(
    orders: pd.DataFrame,
    actuals_by_mlb: Dict[str, pd.Series],
    cutoff: pd.Timestamp,
    horizon: int,
    max_series: int,
    skip_current: bool,
) -> pd.DataFrame:
    """Forecast and collect actuals for one cutoff; returns long-format rows."""
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

    forecasts: Dict[str, Dict[str, pd.Series]] = {
        name: {mlb: fn(histories[mlb], cutoff, horizon) for mlb in eligible}
        for name, fn in BASELINES.items()
    }
    if not skip_current:
        start = time.time()
        forecasts[CURRENT_MODEL_NAME] = forecast_current_model(
            features, sorted(common), cutoff, horizon
        )
        logger.info(
            f"Production model: {len(common)} series in {time.time() - start:.0f}s"
        )

    rows = []
    for mlb in eligible:
        actual = actuals_by_mlb[mlb].reindex(dates, fill_value=0).to_numpy()
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
                    }
                )
            )
    logger.info(
        f"Cutoff {cutoff.date()}: {len(eligible)} eligible series, "
        f"{len(common)} in common set"
    )
    return pd.concat(rows, ignore_index=True)


def run_source(name: str, orders: pd.DataFrame, args: argparse.Namespace) -> Path:
    """Backtest one data source and write its outputs."""
    actuals = build_actuals_grid(orders)
    actuals_by_mlb = {
        mlb: frame.set_index("date")["y"] for mlb, frame in actuals.groupby("mlb")
    }
    cutoffs = make_cutoffs(
        orders["date"].min(),
        orders["date"].max(),
        args.horizon,
        args.folds,
        args.step_days,
    )
    logger.info(f"[{name}] cutoffs: {[c.date().isoformat() for c in cutoffs]}")

    results = pd.concat(
        [
            run_fold(
                orders,
                actuals_by_mlb,
                c,
                args.horizon,
                args.max_series,
                args.skip_current,
            )
            for c in cutoffs
        ],
        ignore_index=True,
    )

    common_scores = summarise(results[results["in_common_set"]]).assign(
        scope=f"common set (top {args.max_series})"
    )
    baseline_rows = results[results["model"].isin(BASELINES)]
    all_scores = summarise(baseline_rows).assign(scope="all eligible series")
    scores = pd.concat([common_scores, all_scores], ignore_index=True)

    out_dir = (
        PROJECT_ROOT / "bld" / "backtest" / f"{name}_{datetime.now():%Y%m%d_%H%M%S}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_parquet(out_dir / "forecasts.parquet", index=False)
    scores.to_csv(out_dir / "scores.csv", index=False)
    summary = render_summary(name, cutoffs, scores, args)
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
    for scope, frame in scores.groupby("scope", sort=False):
        overall = frame[frame["group_type"] == "overall"].sort_values("wape")
        lines += [
            f"## Leaderboard: {scope}",
            "",
            _table(overall, ["model", "wape", "mase", "bias", "series_x_folds"]),
            "",
        ]
        for group_type, title in [
            ("horizon", "WAPE by horizon"),
            ("volume", "WAPE by volume tier"),
        ]:
            pivot = (
                frame[frame["group_type"] == group_type]
                .pivot(index="model", columns="group", values="wape")
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
        "--source", choices=["real", "synthetic", "both"], default="both"
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DATA_DIR / "raw_sql.csv",
        help="Historical order export (default: data/raw_sql.csv)",
    )
    parser.add_argument(
        "--synthetic-csv",
        type=Path,
        default=DATA_DIR / "synthetic_orders.csv",
        help="Synthetic orders; generated with the default seed if missing",
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
        "--skip-current", action="store_true", help="Score baselines only (fast)"
    )
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
        if not args.csv.exists():
            logger.error(f"Historical export not found: {args.csv}")
            return 1
        sources["real"] = load_orders(args.csv)
    if args.source in ("synthetic", "both"):
        if not args.synthetic_csv.exists():
            logger.info("Synthetic data not found; generating with the default seed")
            orders, truth = generate_synthetic_orders()
            save_synthetic(orders, truth, args.synthetic_csv.parent)
        sources["synthetic"] = load_orders(args.synthetic_csv)

    for name, orders in sources.items():
        out_dir = run_source(name, orders, args)
        logger.info(f"[{name}] results written to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
