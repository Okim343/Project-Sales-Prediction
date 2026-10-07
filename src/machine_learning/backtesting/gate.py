"""Pre-switch gate for a fresh MLB-level, common-set backtest.

Usage: python backtesting/gate.py bld/backtest/db_<timestamp>
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def evaluate_gate(scores: pd.DataFrame) -> tuple[bool, list[str]]:
    """Apply the predeclared accuracy and bias criteria to the common set."""
    common = scores[scores["scope"].str.startswith("common set")]
    names = ("lgbm_direct", "current_xgboost", "weekday_mean_4w")
    lines = []

    def one(model, group_type, group):
        selected = common[
            (common["model"] == model)
            & (common["group_type"] == group_type)
            & (common["group"] == group)
        ]
        if len(selected) != 1:
            raise ValueError(
                f"Expected exactly one common-set {model}/{group_type}/{group} score"
            )
        row = selected.iloc[0]
        if not np.isfinite(row["wape"]) or not np.isfinite(row["bias"]):
            raise ValueError(
                f"Non-finite common-set {model}/{group_type}/{group} score"
            )
        return row

    overall = {name: one(name, "overall", "all") for name in names}
    win = overall["lgbm_direct"]["wape"] < min(
        overall["current_xgboost"]["wape"], overall["weekday_mean_4w"]["wape"]
    )
    lines.append(
        f"Overall WAPE below XGBoost and weekday mean: {'PASS' if win else 'FAIL'}"
    )
    bias_ok = abs(overall["lgbm_direct"]["bias"]) <= 0.15
    lines.append(f"Overall bias within ±0.15: {'PASS' if bias_ok else 'FAIL'}")
    groups = common[
        (common["model"] == "current_xgboost")
        & common["group_type"].isin(["horizon", "volume"])
    ][["group_type", "group"]].drop_duplicates()
    if not {"horizon", "volume"}.issubset(set(groups["group_type"])):
        raise ValueError(
            "Missing horizon or volume groups in common-set XGBoost scores"
        )
    group_ok = True
    for group_type, group in groups.itertuples(index=False, name=None):
        lgbm = one("lgbm_direct", group_type, group)["wape"]
        xgb = one("current_xgboost", group_type, group)["wape"]
        passed = lgbm <= xgb + 0.02
        group_ok &= passed
        lines.append(
            f"{group_type}/{group}: {'PASS' if passed else 'FAIL'} ({lgbm:.3f} vs {xgb:.3f})"
        )
    return bool(win and bias_ok and group_ok), lines


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)
    scores = pd.read_csv(args.run_dir / "scores.csv")
    try:
        passed, lines = evaluate_gate(scores)
    except (ValueError, KeyError) as exc:
        print(f"FAIL: {exc}")
        return 1
    print("\n".join(lines))
    print("PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
