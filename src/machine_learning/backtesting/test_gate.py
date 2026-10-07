"""Offline tests for the fixed production switch gate."""

import pandas as pd

from backtesting.gate import evaluate_gate, main


def scores():
    rows = []
    for group_type, group in (
        ("overall", "all"),
        ("horizon", "days 1-7"),
        ("horizon", "days 8-30"),
        ("horizon", "days 31+"),
        ("volume", "top 20%"),
        ("volume", "rest"),
    ):
        for model, wape, bias in (
            ("lgbm_direct", 0.74, -0.1),
            ("current_xgboost", 0.88, 0.08),
            ("weekday_mean_4w", 0.85, 0.1),
        ):
            rows.append(
                dict(
                    model=model,
                    group_type=group_type,
                    group=group,
                    wape=wape,
                    bias=bias,
                    scope="common set (top 100)",
                )
            )
    return pd.DataFrame(rows)


def test_gate_passes_and_cli_reads_scores(tmp_path):
    frame = scores()
    assert evaluate_gate(frame)[0]
    frame.to_csv(tmp_path / "scores.csv", index=False)
    assert main([str(tmp_path)]) == 0


def test_gate_rejects_each_predeclared_failure(tmp_path):
    for model, group_type, group, column, value in (
        ("lgbm_direct", "overall", "all", "wape", 0.9),
        ("lgbm_direct", "horizon", "days 1-7", "wape", 0.91),
        ("lgbm_direct", "volume", "rest", "wape", 0.91),
        ("lgbm_direct", "overall", "all", "bias", -0.16),
    ):
        frame = scores()
        mask = (
            (frame.model == model)
            & (frame.group_type == group_type)
            & (frame.group == group)
        )
        frame.loc[mask, column] = value
        assert not evaluate_gate(frame)[0]
        frame.to_csv(tmp_path / "scores.csv", index=False)
        assert main([str(tmp_path)]) == 1
