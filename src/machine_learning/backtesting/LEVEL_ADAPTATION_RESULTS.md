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

## Part 2: window totals for replenishment

### Pre-registration (written and committed before any run)

Part 1 showed that the long-horizon blend `lgbm_direct_wm4w_long` reduces the level error without improving daily unit WAPE. Replenishment uses demand summed over a protection window, where daily noise partly cancels and the level error does not. Part 2 tests whether the blend improves those window totals.

**Windows.** Protection windows start the day after the cutoff. The blend changes only days 31–90, so on windows of 30 days or less it is identical to `lgbm_direct` by construction. Windows of **60 and 90 days** are the test. Windows of 7, 14 and 28 days are run as controls and must come out identical. Service levels stay at the placeholder values 0.5, 0.8, 0.9 and 0.95. The quantile method is `nb_tier` (negative binomial spread fitted on three inner origins), as in `REPLENISHMENT_RESULTS.md`.

**Runs** (same data, cutoffs and common set as Part 1):

```bash
python backtesting/run_backtest.py --source vendas --end-date 2025-12-31 --tag LW_A \
  --models "weekday_mean_4w,lgbm_direct,lgbm_direct_wm4w_long" \
  --replenishment --windows 7,14,28,60,90 --quantile-methods nb_tier
python backtesting/run_backtest.py --source vendas --end-date 2026-10-31 --tag LW_B \
  --models "weekday_mean_4w,lgbm_direct,lgbm_direct_wm4w_long" \
  --replenishment --windows 7,14,28,60,90 --quantile-methods nb_tier
```

**Decision rule.** The blend **passes** if, in regime A on the common set and on observed sales, all four hold at **both** L = 60 and L = 90, compared with `lgbm_direct`:

1. Window-total WAPE is lower.
2. The absolute window-total bias is smaller.
3. Mean scaled pinball over the four service levels is lower.
4. Window-total WAPE is also lower without AWR1-110X74 (computed from `window_forecasts.parquet`).

It is **recommended** only if its regime B window-total WAPE at L = 90 is no more than 0.010 worse than `lgbm_direct`'s. Coverage at 0.9, fill rate and excess units are reported for context but do not decide. Real outcomes are observed sales and may be censored by stockouts.

### Results

Outputs: `bld/backtest/vendas_LW_A_20261103_154013/` and `bld/backtest/vendas_LW_B_20261103_154506/` (`replenishment_scores.csv`, `window_forecasts.parquet`). Wall time was 282 s for A and 328 s for B; the replenishment step took 3.4 and 4.0 minutes. Cutoffs and common set are unchanged.

**Controls.** At L = 7, 14 and 28 the blend and `lgbm_direct` are identical on every score, as expected.

Below, window WAPE and bias are for the window-total mean. Pinball is the mean scaled pinball over the four service levels. Coverage, fill rate and excess units are at service level 0.9, with `nb_tier` on observed sales.

#### Regime A (decides), common set

| L | Model | Window WAPE | Window bias | Pinball | WAPE without AWR1-110X74 | Coverage 0.9 | Fill rate 0.9 | Excess units 0.9 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 28 | lgbm_direct / blend (identical) | 0.262 | −6.4% | 0.086 | 0.363 | 0.823 | 0.971 | 0.411 |
| 28 | weekday_mean_4w | 0.319 | −2.5% | 0.110 | 0.444 | 0.828 | 0.979 | 0.653 |
| 60 | lgbm_direct | 0.3027 | −11.2% | 0.0951 | 0.4157 | 0.785 | 0.956 | 0.313 |
| 60 | **lgbm_direct_wm4w_long** | **0.2914** | −9.3% | **0.0902** | **0.4099** | 0.803 | 0.965 | 0.358 |
| 60 | weekday_mean_4w | 0.330 | −4.9% | 0.116 | 0.467 | 0.783 | 0.972 | 0.705 |
| 90 | lgbm_direct | 0.3713 | −16.9% | 0.1154 | 0.4818 | 0.778 | 0.946 | 0.264 |
| 90 | **lgbm_direct_wm4w_long** | **0.3465** | −12.8% | **0.1025** | **0.4763** | 0.798 | 0.960 | 0.366 |
| 90 | weekday_mean_4w | 0.355 | −7.1% | 0.131 | 0.516 | 0.798 | 0.966 | 0.780 |

| Rule (blend vs `lgbm_direct`) | L = 60 | L = 90 |
|---|---|---|
| 1. Lower window WAPE | ✓ 0.291 vs 0.303 | ✓ 0.347 vs 0.371 |
| 2. Smaller absolute window bias | ✓ 9.3% vs 11.2% | ✓ 12.8% vs 16.9% |
| 3. Lower mean scaled pinball | ✓ 0.090 vs 0.095 | ✓ 0.103 vs 0.115 |
| 4. Lower window WAPE without AWR1-110X74 | ✓ 0.410 vs 0.416 | ✓ 0.476 vs 0.482 |

By cutoff, at L = 90 the blend has a lower window WAPE at three of four cutoffs; it loses 2025-07-04 (0.426 vs 0.415). By tier it is better in both (rest 0.624 vs 0.636; top 20% 0.289 vs 0.316).

#### Regime B (check), common set

| L | Model | Window WAPE | Window bias | Pinball | WAPE without AWR1-110X74 | Coverage 0.9 | Fill rate 0.9 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 28 | lgbm_direct / blend (identical) | 0.329 | −0.4% | 0.106 | 0.319 | 0.809 | 0.960 |
| 28 | weekday_mean_4w | 0.326 | +3.2% | 0.118 | 0.336 | 0.801 | 0.947 |
| 60 | lgbm_direct | **0.2812** | +5.8% | **0.1007** | 0.3244 | 0.783 | 0.964 |
| 60 | lgbm_direct_wm4w_long | 0.2852 | +5.9% | 0.1011 | **0.3208** | 0.783 | 0.966 |
| 60 | weekday_mean_4w | 0.326 | +7.8% | 0.126 | 0.346 | 0.783 | 0.945 |
| 90 | lgbm_direct | **0.3277** | +9.1% | **0.1156** | 0.3611 | 0.773 | 0.959 |
| 90 | lgbm_direct_wm4w_long | 0.3305 | +8.8% | 0.1208 | **0.3527** | 0.768 | 0.949 |
| 90 | weekday_mean_4w | 0.373 | +9.8% | 0.144 | 0.377 | 0.768 | 0.931 |

The regime B check passes: at L = 90 the blend's window WAPE is 0.003 worse than `lgbm_direct`'s, within the 0.010 allowed.

### Verdict: the blend **passes** and is **recommended** under the pre-registered rule

On 60- and 90-day window totals in regime A, `lgbm_direct_wm4w_long` is more accurate than `lgbm_direct`: window WAPE −0.011 at 60 days and −0.025 at 90 days. It is less biased and has lower pinball, and it still wins without the ribbon. In regime B it is a wash: slightly worse on all series and on pinball, slightly better without the ribbon. Windows of 30 days or less are unaffected.

### Caveats

- **This was not an independent test.** The blend was chosen after seeing regime A in Part 1, and Part 2 scored the same regime A cutoffs. Regime B is the only data the choice did not influence, and there the blend is neutral rather than better. The evidence supports "it helps in a growing market and does no harm after a break". It does not show that the blend is better in general.
- **It matters only for windows longer than 30 days.** If the client's protection window (lead time plus review period) is 30 days or less, the blend changes nothing.
- **Coverage is too low for every model.** At service level 0.9, actual coverage is 0.77–0.80 at 60–90 days, and only 0.70–0.74 for the top tier under `lgbm_direct` and the blend in regime A. The NB spread fitted on three inner origins underestimates window uncertainty at long windows. This is a bigger replenishment issue than the mean model and should be fixed before any service level is promised.
- **Observed sales only.** Outcomes may be censored by stockouts, and service levels and windows are still placeholders pending business inputs.
- **Production would need a code change.** Adopting the blend means computing `weekday_mean_4w` and blending days 31–90 in `estimation/lgbm_forecast.py`. That change has not been made.
