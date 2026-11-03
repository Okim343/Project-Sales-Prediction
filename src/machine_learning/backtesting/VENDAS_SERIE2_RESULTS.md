# Frozen `lgbm_direct` on Greycom invoice sales (NF-e series 2)

This is an evaluation only: the model, its features and its hyperparameters are unchanged. The production model `lgbm_direct` (global Tweedie LightGBM, power 1.5, direct blocks 1–7 / 8–30 / 31–90, `month` dropped, stockout masking on, 300 trees, seed 42, no calibration) is compared with the three baselines. The rolling-origin harness uses 4 cutoffs, 30 days apart, a 90-day horizon and the production eligibility rule. The common set is the top 100 eligible series by training volume. `current_xgboost` was not run.

Runs (from `src/machine_learning`):

```bash
python backtesting/run_backtest.py --source vendas --end-date 2025-12-31 --tag A --models "baselines,lgbm_direct"
python backtesting/run_backtest.py --source vendas --end-date 2026-10-31 --tag B --models "baselines,lgbm_direct"
```

Outputs: `bld/backtest/vendas_A_20261103_142506/` and `bld/backtest/vendas_B_20261103_142628/`. Wall time was 76 s for A and 82 s for B, including loading. `lgbm_direct` took about 15 s per fold.

## Data

`data/vendas_ml_serie2_2026-11-01/vendas_sku_dia_serie2.csv` has 119,942 SKU × day rows from 17 Dec 2021 to 1 Nov 2026, without cancelled invoices. It is loaded by `backtesting/vendas_data.py::load_vendas_orders`, which builds one pseudo-order per SKU-day with gross units sold:

- `mlb` = SKU
- quantity = `unidades_vendidas`
- unit price = `valor_produtos / unidades_vendidas`

The harness aggregates to days anyway, so this is equivalent to the item file.

- **Dates:** invoice-issue days in Brasília time, read as naive dates. The loader checks that normalisation leaves every date on its calendar day.
- **Last day:** 1 Nov 2026 is incomplete and dropped (57 rows, 1,350 units).
- **Return-only rows:** 1,394 SKU-days with zero gross units (returns only) are dropped.

### Reference-figure check (whole file, before exclusions)

`LEIAME.md` reports **items sold** (`itens_vendidos`, invoice lines), not units. Items and net revenue (`receita_liquida`) match to rounding (< R$ 1) in every year. Units are listed for context; they are not comparable with the reference.

| Year | Units | Items | Items (ref) | Net revenue R$ | Revenue (ref) |
|---|---:|---:|---:|---:|---:|
| 2021 | 116 | 108 | 108 | 23,737 | 23,737 |
| 2022 | 681,321 | 74,100 | 74,100 | 9,335,376 | 9,335,376 |
| 2023 | 735,069 | 122,630 | 122,630 | 25,444,358 | 25,444,358 |
| 2024 | 1,132,002 | 313,100 | 313,100 | 60,073,402 | 60,073,402 |
| 2025 | 1,048,742 | 409,189 | 409,189 | 82,815,821 | 82,815,821 |
| 2026 (to 1 Nov) | 14,190,982 | 932,439 | 932,439 | 184,286,835 | 184,286,835 |

### Exclusions

The 11 World Cup sticker and album SKUs in `WORLD_CUP_SKUS` are removed from training and scoring.

| Regime | Rows removed | Units removed | Units before removal |
|---|---:|---:|---:|
| A (to 2025-12-31) | 886 | 533,505 | 3,597,250 |
| B (to 2026-10-31) | 1,645 | 13,613,961 | 17,786,882 |

In 2026 (to 31 Oct) these SKUs are 92.2% of units but only 40.9% of invoiced value. The description scan (`COPA|FIFA|ALBUM|FIGURINHA`) found one further match, which was reported and kept as instructed: `ENVELOPEFIFACLUBE25` (FIFA Club World Cup 25 packets, 970 units on 8 days). It is never eligible, so it does not affect scores.

## Cutoffs, eligibility and concentration

| Regime | Cutoffs | Eligible series per cutoff | Common set |
|---|---|---|---|
| A | 2025-07-04, 08-03, 09-02, 10-02 | 133, 126, 116, 126 | 100 each |
| B | 2026-05-04, 06-03, 07-03, 08-02 | 112, 99, 98, 107 | 100, 99, 98, 100 |

Both lists match the expected cutoffs. Only about half of the SKUs selling in each window are eligible. The common set covers 98–99% of the eligible series' scored units and value, so the two scopes give nearly the same scores.

How much of the store's (non-World Cup) sales the eligible series cover in each scoring window:

| Cutoff | SKUs selling in window | Eligible | Units covered | Value covered | Value from SKUs launched after the cutoff |
|---|---:|---:|---:|---:|---:|
| 2025-07-04 | 259 | 133 | 89% | 83% | 5% |
| 2025-08-03 | 264 | 126 | 90% | 83% | 3% |
| 2025-09-02 | 250 | 116 | 91% | 89% | 1% |
| 2025-10-02 | 240 | 126 | 92% | 90% | 0% |
| 2026-05-04 | 147 | 112 | 98% | 94% | 5% |
| 2026-06-03 | 156 | 99 | 94% | 80% | 19% |
| 2026-07-03 | 153 | 98 | 88% | 65% | 34% |
| 2026-08-02 | 148 | 107 | 84% | 51% | 24% |

**Unit concentration.** Top 5 SKUs' share of scored actual units, common set; the all-eligible figures differ by < 0.01. The value share uses the cutoff price.

| Regime A | Unit share | Value share | | Regime B | Unit share | Value share |
|---|---:|---:|---|---|---:|---:|
| AWR1-110X74 (Zebra ribbon) | 0.498 | 0.029 | | AWR1-110X74 | 0.387 | 0.023 |
| 100X150T27,0M (label roll) | 0.140 | 0.024 | | 100X150T27,0M | 0.142 | 0.024 |
| MO300 (mouse) | 0.037 | 0.004 | | 40X40X1TX20M (label roll) | 0.123 | 0.008 |
| TC213 (keyboard) | 0.029 | 0.010 | | GPCHOICEGAFC10,1KG (cat food) | 0.025 | 0.041 |
| GPCHOICEGAFC10,1KG (cat food) | 0.017 | 0.029 | | GPCHOICEAFC10,1KG (cat food) | 0.014 | 0.022 |
| **Top 5** | **0.721** | **0.096** | | **Top 5** | **0.691** | **0.118** |

The concentration runs the other way in value. In regime A, 12 SKUs priced at R$ 300 or more hold 56% of scored value but 5% of units. They are thermal and label printers (Elgin L42 Pro Full, L42DT and I9 Full, Bematech MP4200), the Elgin EL8600 barcode reader and the Prix 4 Uno scale.

## Metric

`wape_value` = Σ pᵢ·|forecast − actual| / Σ pᵢ·actual. Here pᵢ is series i's last known unit price on or before the cutoff: revenue / units on its last sale day ≤ cutoff (`data_prep.last_unit_price`). It is a scoring weight only; the model never sees it, and no future price is used. `metrics.summarise` adds it only when a `price_weight` column is present, so other sources are unchanged. Bias > 0 means over-forecasting. MASE uses the in-sample 7-day seasonal-naive MAE.

## Regime A (stable 2025, decides)

### Leaderboards

Common set (top 100), 400 series × folds:

| Model | Unit WAPE | Revenue-weighted WAPE | MASE | Bias |
|---|---:|---:|---:|---:|
| lgbm_direct | **0.544** | 0.669 | **1.115** | −16.9% |
| weekday_mean_4w | 0.588 | **0.650** | 1.168 | −7.1% |
| moving_average_28 | 0.610 | 0.667 | 1.160 | −7.2% |
| seasonal_naive_7 | 0.652 | 0.695 | 1.235 | −6.0% |

All eligible series, 501 series × folds:

| Model | Unit WAPE | Revenue-weighted WAPE | MASE | Bias |
|---|---:|---:|---:|---:|
| lgbm_direct | **0.551** | 0.679 | **1.086** | −16.7% |
| weekday_mean_4w | 0.595 | **0.661** | 1.141 | −6.8% |
| moving_average_28 | 0.616 | 0.677 | 1.129 | −6.8% |
| seasonal_naive_7 | 0.659 | 0.708 | 1.211 | −5.7% |

### Breakdowns (common set; unit WAPE / revenue-weighted WAPE)

| Model | Days 1–7 | Days 8–30 | Days 31+ | Top 20% | Rest |
|---|---|---|---|---|---|
| lgbm_direct | 0.429 / 0.513 | 0.501 / 0.613 | 0.573 / 0.701 | 0.471 / 0.543 | 0.897 / 0.797 |
| weekday_mean_4w | 0.519 / 0.598 | 0.563 / 0.617 | 0.605 / 0.666 | 0.506 / 0.509 | 0.980 / 0.794 |
| moving_average_28 | 0.539 / 0.644 | 0.588 / 0.639 | 0.626 / 0.679 | 0.536 / 0.551 | 0.964 / 0.787 |
| seasonal_naive_7 | 0.521 / 0.621 | 0.642 / 0.683 | 0.671 / 0.706 | 0.572 / 0.571 | 1.033 / 0.822 |

| Model | 2025-07-04 | 2025-08-03 | 2025-09-02 | 2025-10-02 |
|---|---|---|---|---|
| lgbm_direct, unit / value WAPE | 0.588 / 0.783 | 0.511 / 0.643 | 0.535 / 0.650 | 0.544 / 0.631 |
| weekday_mean_4w, unit / value WAPE | 0.655 / 0.816 | 0.575 / 0.662 | 0.530 / 0.589 | 0.598 / 0.593 |
| lgbm_direct bias | −9.3% | −14.2% | −19.7% | −22.9% |
| weekday_mean_4w bias | −16.0% | +5.0% | −12.7% | −4.5% |

The bias of `lgbm_direct` by horizon is −8% (days 1–7), −7% (days 8–30) and −21% (days 31+). By tier it is −18% (top 20%) and −11% (rest). The all-eligible breakdowns are within 0.02 of these and are in `summary.md`.

Wins and losses of `lgbm_direct` against `weekday_mean_4w`:

- **Unit WAPE:** wins overall, at every horizon, in both tiers, and at 3 of 4 cutoffs. It loses 2025-09-02 narrowly (0.535 vs 0.530).
- **Revenue-weighted WAPE:** wins days 1–7 and 8–30 and the July and August cutoffs. It loses days 31+, both tiers (rest by 0.003) and the September and October cutoffs.

### Verdict: **FAIL** under the pre-registered rule

The rule requires both unit WAPE and revenue-weighted WAPE to be lower than `weekday_mean_4w`'s on the common set:

- Unit WAPE passes: 0.544 vs 0.588 (−7.5%).
- Revenue-weighted WAPE fails: 0.669 vs 0.650 (+2.9%).

MASE favours `lgbm_direct` (1.115 vs 1.168).

**Why the two WAPEs disagree.** The unit win comes from the cheap, high-volume SKUs:

- AWR1-110X74 alone accounts for 1.3 points of the 4.3-point unit-WAPE gain (WAPE 0.382 vs 0.408).
- Next come GB9786555116762 (Bobbie Goods book), PD587/PD589 (pen drives), MO300 and TC213.

The value loss comes from the high-price printers and readers. `lgbm_direct` under-forecasts them far more than the baseline:

| SKU (cutoff price) | Actual units, 4 folds | lgbm_direct forecast | weekday_mean_4w forecast |
|---|---:|---:|---:|
| L42PROFULL (≈ R$ 1,300) | 12,464 | 7,618 | 10,319 |
| I9FULL (≈ R$ 710) | 8,599 | 1,089 | 3,365 |
| L42DT (≈ R$ 970) | 4,368 | 1,987 | 3,250 |
| MP4200 (≈ R$ 620) | 11,249 | 4,685 | 6,489 |
| EL8600 (≈ R$ 1,050) | 3,496 | 567 | 1,341 |

On SKUs priced at R$ 300 or more:

- `lgbm_direct`: revenue-weighted WAPE 0.630, bias −47%
- `weekday_mean_4w`: revenue-weighted WAPE 0.564, bias −35%

On cheaper SKUs `lgbm_direct` wins on value too: 0.718 vs 0.760. These printers ramped up through autumn 2025; store revenue rose from R$ 6.2 M in August to R$ 11.1 M in November. The global model shrinks their level toward the panel more than a 4-week mean does.

**Bias.** The overall bias is −16.9%, against about −10.8% on the old SKU export, and it worsens with the cutoff (−9% → −23%) and the horizon (days 31+: −21%). The October cutoff's horizon covers Black Friday 2025. For that cutoff the common set sold 98.4k units in November and 96.8k in December; `lgbm_direct` forecast 73.8k and 63.1k (`weekday_mean_4w`: 85.8k and 92.6k). The known under-forecasting is larger here, and it grows in a rising market.

## Regime B (2026 stress test, no pass/fail)

### Leaderboards

Common set (top 100), 397 series × folds:

| Model | Unit WAPE | Revenue-weighted WAPE | MASE | Bias |
|---|---:|---:|---:|---:|
| lgbm_direct | **0.621** | **0.735** | **1.493** | +9.1% |
| weekday_mean_4w | 0.630 | 0.762 | 1.521 | +9.8% |
| moving_average_28 | 0.684 | 0.777 | 1.518 | +9.8% |
| seasonal_naive_7 | 0.687 | 0.794 | 1.626 | −2.3% |

All eligible series, 416 series × folds:

| Model | Unit WAPE | Revenue-weighted WAPE | MASE | Bias |
|---|---:|---:|---:|---:|
| lgbm_direct | **0.622** | **0.737** | **1.494** | +9.0% |
| weekday_mean_4w | 0.631 | 0.763 | 1.518 | +9.7% |
| moving_average_28 | 0.684 | 0.778 | 1.513 | +9.7% |
| seasonal_naive_7 | 0.687 | 0.796 | 1.626 | −2.4% |

### Breakdowns (common set; unit WAPE / revenue-weighted WAPE)

| Model | Days 1–7 | Days 8–30 | Days 31+ | Top 20% | Rest |
|---|---|---|---|---|---|
| lgbm_direct | 0.439 / 0.540 | 0.592 / 0.634 | 0.658 / 0.803 | 0.588 / 0.724 | 0.734 / 0.747 |
| weekday_mean_4w | 0.450 / 0.613 | 0.613 / 0.673 | 0.662 / 0.817 | 0.594 / 0.757 | 0.752 / 0.767 |
| moving_average_28 | 0.533 / 0.631 | 0.661 / 0.684 | 0.714 / 0.834 | 0.666 / 0.791 | 0.745 / 0.762 |
| seasonal_naive_7 | 0.516 / 0.615 | 0.679 / 0.685 | 0.714 / 0.862 | 0.649 / 0.760 | 0.814 / 0.829 |

| Model | 2026-05-04 | 2026-06-03 | 2026-07-03 | 2026-08-02 |
|---|---|---|---|---|
| lgbm_direct, unit / value WAPE | 0.606 / 0.637 | 0.591 / 0.676 | 0.684 / 0.789 | 0.608 / 0.869 |
| weekday_mean_4w, unit / value WAPE | 0.563 / 0.673 | 0.568 / 0.642 | 0.707 / 0.843 | 0.692 / 0.925 |
| lgbm_direct bias | +12.1% | +6.5% | +21.1% | −2.6% |
| weekday_mean_4w bias | +13.0% | +8.9% | +26.7% | −8.6% |

The bias of `lgbm_direct` by horizon is −11% (days 1–7), +5% (days 8–30) and +14% (days 31+). By tier it is +18% (top 20%) and −23% (rest).

### Where the model breaks

- **Everyone does worse than in 2025.** MASE is about 1.5 for all models, against about 1.1–1.2 in regime A. The 90-day forecasts are worse than an in-sample weekly naive, and value WAPE reaches 0.87–0.93 at the last cutoff. `lgbm_direct` still ranks first overall on all three metrics, but its edge on units is small (0.621 vs 0.630).
- **By cutoff.** `lgbm_direct` loses unit WAPE at the May and June cutoffs (0.606 vs 0.563, 0.591 vs 0.568). At June it also loses value WAPE: its days 1–7 bias is −35%, against −15% for the baseline. The July cutoff is the worst for everyone. Days 8–30 WAPE is 0.81, with a +39% bias: the scored SKUs dipped in July 2026 (82.6k units actual against about 106k forecast). The dip coincides with the World Cup (11 Jun – 19 Jul).
- **By horizon.** Over-forecasting grows with the horizon (+14% at days 31+). This is the reverse of regime A: the model carries forward the pre-break level of the established SKUs, which then flattened.
- **By tier.** The top 20% are over-forecast (+18%). The rest tier stays under-forecast (−23%), as in the old export.
- **SKU churn.** I9FULL (Elgin I9 printer) was relisted as I9FULL2 from late June 2026. I9FULL fell from 1,328 units in April to 3 in September. Both models over-forecast it heavily (WAPE 5.4 for `lgbm_direct`, 3.9 for the baseline), and it is the largest value loss for `lgbm_direct` in B. Other over-forecast SKUs are FLASH (WAPE 3.7) and QGGCDDM10.1KG (1.2). The model wins clearly on the Quatree cat-food line (QSGCF10.1KG: 1.41 vs 2.27).
- **New SKUs launched after May 2026 are not tested at all.** 38 SKUs first sold on or after 1 May 2026. None is eligible at any cutoff, because the rule needs horizon + 15 = 105 cleaned rows. They earned R$ 21.5 M of 2026 non-World Cup invoiced value (R$ 108.8 M), led by:
  - Bambu Lab A1 3D printers: R$ 9.1 M from 17 Aug
  - a new FN pet-food line from mid-July: R$ 1.5 M, R$ 1.0 M, R$ 0.8 M…
  - I9FULL2: R$ 0.7 M

  As a result the eligible series cover only 51% of the value in the last cutoff's window, and 65% in the July window. The May growth break the regime was meant to stress is mostly a new-SKU effect, and the harness cannot see it. The scored SKUs sell about 80–110k units a month, about the same as in late 2025.

## Caveats

- **SKU level, not MLB level.** The series are SKUs, while production forecasts per MLB (listing). One SKU can have several listings, and relistings such as I9FULL → I9FULL2 split a product's history.
- **Invoice dates.** The date is when the invoice was issued (Full: at picking, usually the purchase day or the next; own stock: at dispatch), not the order date.
- **Gross sales.** The target is gross units sold. Returns and undelivered goods are not netted out, because they are dated when the goods came back.
- **No stock data.** Zero days may be stockouts. The model's stockout proxy was left as is, and the scored actuals are censored sales.
- **Dropped data.** 1 Nov 2026 is incomplete and was dropped. The data ends on 31 Oct 2026 (regime B) or on 31 Dec 2025 (regime A).
- **Revenue weights.** They are cutoff unit prices from invoiced values (after promotions, without ML fees or freight). A product that changes price inside the horizon is weighted at its cutoff price.
- **Units across SKUs.** Unit metrics add rolls, ribbons and printers as equal units. Read unit WAPE together with the concentration and value tables.
