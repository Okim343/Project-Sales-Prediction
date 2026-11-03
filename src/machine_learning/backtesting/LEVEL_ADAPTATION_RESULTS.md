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

Outputs: `bld/backtest/vendas_LA_A_20261103_152039/` and `bld/backtest/vendas_LA_B_20261103_152350/`. Wall time was 185 s for A and 192 s for B. Each LightGBM variant took 13–15 s per fold. `lgbm_direct` reproduces the scores in `VENDAS_SERIE2_RESULTS.md` exactly. Cutoffs, eligible series and the common set are unchanged.

### Regime A (decides), common set

| Model | Unit WAPE | WAPE days 1–7 | WAPE days 31+ | Bias days 31+ | Bias overall | Unit WAPE without AWR1-110X74 | MASE |
|---|---:|---:|---:|---:|---:|---:|---:|
| lgbm_direct | 0.5443 | 0.4286 | 0.5734 | −21.2% | −16.9% | **0.7053** | 1.115 |
| lgbm_direct_train180 | 0.5676 | 0.4512 | 0.6016 | −21.8% | −19.0% | 0.7242 | 1.119 |
| lgbm_direct_scaled_28 | 0.5713 | 0.4433 | 0.6073 | −15.3% | −13.5% | 0.7122 | 1.135 |
| lgbm_direct_wm4w_long | **0.5432** | 0.4286 | **0.5717** | **−15.2%** | **−12.8%** | 0.7091 | **1.104** |
| weekday_mean_4w | 0.588 | 0.519 | 0.605 | −9.3% | −7.1% | 0.767 | 1.168 |

| Rule | train180 | scaled_28 | wm4w_long |
|---|---|---|---|
| 1. Lower overall unit WAPE | ✗ 0.568 | ✗ 0.571 | ✓ 0.5432 vs 0.5443 |
| 2. Smaller absolute bias on days 31+ | ✗ 21.8% | ✓ 15.3% | ✓ 15.2% |
| 3. Days 1–7 no more than 0.005 worse | ✗ +0.023 | ✗ +0.015 | ✓ identical |
| 4. Lower unit WAPE without AWR1-110X74 | ✗ 0.724 | ✗ 0.712 | ✗ 0.709 vs 0.705 |
| **Passes** | no | no | **no** |

### Regime B (check), common set

| Model | Unit WAPE | WAPE days 31+ | Bias days 31+ | Bias overall | Unit WAPE without AWR1-110X74 | MASE |
|---|---:|---:|---:|---:|---:|---:|
| lgbm_direct | 0.621 | 0.658 | +13.6% | +9.1% | 0.628 | 1.493 |
| lgbm_direct_train180 | 0.641 | 0.687 | +8.4% | +5.5% | **0.624** | **1.464** |
| lgbm_direct_scaled_28 | 0.674 | 0.728 | +14.9% | +10.0% | 0.690 | 1.530 |
| lgbm_direct_wm4w_long | **0.613** | **0.646** | +13.2% | +8.8% | 0.630 | 1.473 |
| weekday_mean_4w | 0.630 | 0.662 | +12.7% | +9.8% | 0.663 | 1.521 |

The all-eligible scores are within 0.008 of the common-set scores in both regimes and give the same ranking. They are in each run's `summary.md`.

### Verdict: no candidate passes; `lgbm_direct` stays as it is

- **`lgbm_direct_train180` and `lgbm_direct_scaled_28` are worse on units in regime A.** This holds overall, on days 1–7 and without the ribbon. `scaled_28` reduces the long-horizon bias but loses accuracy everywhere. In regime B both are also worse overall; `train180` is slightly better only without AWR1-110X74 (0.624 vs 0.628).
- **`lgbm_direct_wm4w_long` is the only candidate close to passing.**
  - What it does: it removes about a quarter of the regime A level error (days 31+ bias −21% → −15%, overall bias −17% → −13%), with a marginally better unit WAPE and MASE. In regime B its overall unit WAPE is also lower (0.613 vs 0.621).
  - Why it fails: the unit gain comes from AWR1-110X74 alone. Without that ribbon the blend is worse in both regimes (A: 0.709 vs 0.705; B: 0.630 vs 0.628). It fails condition 4.

### What this shows

- **Removing the level error barely moves unit WAPE.** In regime A the blend cut the overall bias from −17% to −13%, but unit WAPE changed by only 0.001. At the daily level most of the unit error is day-to-day variation around the level, not the level itself. Better level tracking is unlikely to improve daily unit WAPE much.
- **The bias reduction may still matter for window totals.** Replenishment decisions sum forecasts over 7–28 day windows, where day-to-day errors partly cancel and the level error does not. Whether the blend helps there is a separate question; this experiment didn't test it. It could be checked with the existing `--replenishment` harness.
- **Without the ribbon the picture is different.** The two regimes disagree on whether a shorter training window helps the remaining SKUs. That gives no consistent reason to change `train_days`.
