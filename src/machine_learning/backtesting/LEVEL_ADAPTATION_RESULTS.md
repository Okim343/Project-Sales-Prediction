# Level adaptation for `lgbm_direct` on Greycom invoice sales

Follow-up to `VENDAS_SERIE2_RESULTS.md`. That evaluation found that `lgbm_direct` is the best model on units but gets the level wrong at long horizons:

- Regime A (2025): bias −21% on days 31+, worsening across cutoffs from −9% to −23%.
- Regime B (2026): bias +14% on days 31+.

Its level under-reacts in both directions. This experiment tests whether a faster-moving level improves **unit** accuracy. Revenue is not a decision metric here.

## Pre-registration (written and committed before any run)

### Candidates

All are backtest-only. Production code paths (`estimation/`, `pipeline/`, `config.py`) are unchanged.

| Name | Change from `lgbm_direct` | Mechanism |
|---|---|---|
| `lgbm_direct_train180` | `train_days` 180 instead of 365 | Fits on the most recent half year, so older levels weigh less |
| `lgbm_direct_scaled_28` | Existing `target_scale=28`: fit sales divided by the 28-day origin level, then multiply back | Forecasts scale with each series' current level |
| `lgbm_direct_wm4w_long` | Days 1–30 unchanged; days 31–90 = 0.5 × `lgbm_direct` + 0.5 × `weekday_mean_4w` | Anchors the long horizon on the last four weeks |

A recency-weighted variant was considered and dropped: the core model has no recency weight, and adding one would change production code.

### Runs

The same data, exclusions, cutoffs and common set (top 100) as `VENDAS_SERIE2_RESULTS.md`:

```bash
python backtesting/run_backtest.py --source vendas --end-date 2025-12-31 --tag LA_A \
  --models "baselines,lgbm_direct,lgbm_direct_train180,lgbm_direct_scaled_28,lgbm_direct_wm4w_long" \
  --score-without AWR1-110X74
python backtesting/run_backtest.py --source vendas --end-date 2026-10-31 --tag LA_B \
  --models "baselines,lgbm_direct,lgbm_direct_train180,lgbm_direct_scaled_28,lgbm_direct_wm4w_long" \
  --score-without AWR1-110X74
```

### Decision rule

A candidate **passes** if, in regime A on the common set and compared with `lgbm_direct`, all four hold:

1. Overall unit WAPE is lower.
2. The absolute bias on days 31+ is smaller.
3. Unit WAPE on days 1–7 is not worse by more than 0.005.
4. Overall unit WAPE is also lower on the common set without `AWR1-110X74`.

A passing candidate is **recommended** only if its regime B overall unit WAPE (common set) is no more than 0.010 worse than `lgbm_direct`'s. If several are recommended, the one with the lowest regime A unit WAPE is chosen. MASE is reported but does not decide. If no candidate passes, `lgbm_direct` stays as it is.

## Results

Pending.
