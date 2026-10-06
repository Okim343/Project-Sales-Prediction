# Direct LightGBM bias-correction backtest

Updated after fixing validation-series selection. Four 90-day rolling-origin cutoffs per source; common set is up to 100 production-eligible series per outer cutoff. Each score cell is **WAPE / MASE / bias** on observed sales unless marked true demand. Bias is `(forecast − actual) / actual`. True-demand MASE retains the harness’s observed-sales seasonal-naive denominator.

## Validation design

At outer cutoff `c`, the inner model trains only through `c − 90`. Validation listings are chosen by the same production eligibility rule at **`c − 90`** and scored over `c − 89 … c`, including listings that become inactive by `c`. Factor estimation uses observed sales. `calibration_factors.csv` in each run folder records inner and outer cutoffs, model, block, tier, validation totals and series count, raw/applied factor, and whether the floor binds. The raw ratio is clipped to [0.8, 1.5]. The previously promoted variant still uses 35% shrinkage and a 1.06 floor and is now explicitly named `lgbm_direct_cal_shrunk`; `lgbm_direct` is again uncalibrated.

## Common-set score tables

### Real observed sales

| Model | Overall | Days 1–7 | Days 8–30 | Days 31+ | Top 20% | Rest |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Direct (default, uncalibrated) | 0.748 / 1.090 / -0.108 | 0.515 / 0.858 / -0.046 | 0.701 / 1.143 / -0.064 | 0.798 / 1.097 / -0.135 | 0.713 / 1.230 / -0.061 | 0.815 / 1.053 / -0.200 |
| Raw block + tier ratio | 0.803 / 1.179 / +0.099 | 0.520 / 0.867 / +0.024 | 0.746 / 1.229 / +0.063 | 0.862 / 1.196 / +0.123 | 0.771 / 1.352 / +0.163 | 0.863 / 1.133 / -0.024 |
| Shrunk ratio + 1.06 floor | 0.765 / 1.119 / -0.010 | 0.519 / 0.876 / +0.020 | 0.717 / 1.169 / +0.028 | 0.816 / 1.128 / -0.030 | 0.731 / 1.266 / +0.047 | 0.829 / 1.080 / -0.118 |
| Fixed ×1.10 | 0.765 / 1.117 / -0.019 | 0.527 / 0.889 / +0.050 | 0.719 / 1.170 / +0.029 | 0.814 / 1.123 / -0.049 | 0.731 / 1.262 / +0.033 | 0.828 / 1.078 / -0.120 |
| 50/50 direct + recursive | 0.759 / 1.130 / -0.059 | 0.506 / 0.854 / -0.042 | 0.701 / 1.161 / -0.050 | 0.816 / 1.150 / -0.065 | 0.720 / 1.244 / -0.034 | 0.833 / 1.099 / -0.105 |
| Recursive | 0.789 / 1.196 / -0.009 | 0.510 / 0.866 / -0.039 | 0.714 / 1.197 / -0.035 | 0.856 / 1.234 / +0.005 | 0.746 / 1.296 / -0.008 | 0.871 / 1.169 / -0.011 |
| Weekday mean | 0.851 / 1.254 / +0.121 | 0.596 / 0.998 / +0.105 | 0.763 / 1.255 / +0.032 | 0.920 / 1.284 / +0.162 | 0.826 / 1.409 / +0.155 | 0.897 / 1.213 / +0.055 |
| 28-day moving average | 0.910 / 1.297 / +0.121 | 0.679 / 1.056 / +0.105 | 0.829 / 1.310 / +0.031 | 0.974 / 1.321 / +0.163 | 0.910 / 1.566 / +0.156 | 0.912 / 1.227 / +0.056 |
| Seasonal naive | 0.919 / 1.365 / +0.088 | 0.653 / 1.078 / +0.071 | 0.815 / 1.380 / +0.006 | 0.996 / 1.392 / +0.125 | 0.882 / 1.510 / +0.078 | 0.989 / 1.327 / +0.106 |
| Production XGBoost | 0.908 / 1.435 / +0.081 | 0.731 / 1.249 / -0.065 | 0.808 / 1.415 / -0.071 | 0.973 / 1.464 / +0.165 | 0.823 / 1.463 / +0.107 | 1.070 / 1.427 / +0.032 |

### Synthetic observed sales

| Model | Overall | Days 1–7 | Days 8–30 | Days 31+ | Top 20% | Rest |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Direct (default, uncalibrated) | 0.773 / 1.289 / -0.101 | 0.637 / 1.054 / -0.125 | 0.776 / 1.181 / -0.055 | 0.786 / 1.358 / -0.113 | 0.765 / 1.252 / -0.093 | 0.837 / 1.341 / -0.161 |
| Raw block + tier ratio | 0.774 / 1.302 / -0.092 | 0.636 / 1.054 / -0.153 | 0.773 / 1.179 / -0.073 | 0.790 / 1.378 / -0.092 | 0.766 / 1.257 / -0.088 | 0.846 / 1.365 / -0.125 |
| Shrunk ratio + 1.06 floor | 0.787 / 1.314 / -0.046 | 0.646 / 1.072 / -0.072 | 0.793 / 1.207 / +0.001 | 0.800 / 1.384 / -0.060 | 0.779 / 1.280 / -0.039 | 0.849 / 1.363 / -0.109 |
| Fixed ×1.10 | 0.797 / 1.332 / -0.011 | 0.652 / 1.087 / -0.037 | 0.805 / 1.227 / +0.039 | 0.810 / 1.401 / -0.025 | 0.790 / 1.300 / -0.003 | 0.858 / 1.377 / -0.077 |
| 50/50 direct + recursive | 0.773 / 1.289 / -0.079 | 0.634 / 1.049 / -0.113 | 0.767 / 1.173 / -0.049 | 0.790 / 1.362 / -0.086 | 0.766 / 1.257 / -0.071 | 0.835 / 1.335 / -0.143 |
| Recursive | 0.787 / 1.310 / -0.058 | 0.636 / 1.052 / -0.100 | 0.768 / 1.178 / -0.042 | 0.810 / 1.390 / -0.058 | 0.781 / 1.285 / -0.050 | 0.841 / 1.344 / -0.126 |
| Weekday mean | 0.838 / 1.377 / -0.092 | 0.735 / 1.200 / -0.047 | 0.832 / 1.251 / -0.018 | 0.851 / 1.446 / -0.122 | 0.831 / 1.323 / -0.088 | 0.898 / 1.452 / -0.130 |
| 28-day moving average | 0.844 / 1.375 / -0.093 | 0.743 / 1.199 / -0.047 | 0.838 / 1.254 / -0.017 | 0.858 / 1.442 / -0.124 | 0.839 / 1.333 / -0.088 | 0.890 / 1.434 / -0.130 |
| Seasonal naive | 0.926 / 1.545 / -0.160 | 0.760 / 1.305 / -0.118 | 0.935 / 1.443 / -0.092 | 0.940 / 1.612 / -0.187 | 0.915 / 1.504 / -0.162 | 1.012 / 1.603 / -0.138 |
| Production XGBoost | 0.814 / 1.353 / -0.153 | 0.800 / 1.256 / -0.097 | 0.822 / 1.264 / -0.079 | 0.813 / 1.398 / -0.184 | 0.801 / 1.286 / -0.142 | 0.919 / 1.447 / -0.243 |

### Synthetic planted true demand

| Model | Overall | Days 1–7 | Days 8–30 | Days 31+ | Top 20% | Rest |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Direct (default, uncalibrated) | 0.672 / 1.298 / -0.209 | 0.625 / 1.139 / -0.237 | 0.652 / 1.177 / -0.186 | 0.684 / 1.362 / -0.214 | 0.663 / 1.234 / -0.200 | 0.746 / 1.387 / -0.281 |
| Raw block + tier ratio | 0.673 / 1.306 / -0.201 | 0.626 / 1.141 / -0.261 | 0.652 / 1.176 / -0.201 | 0.686 / 1.375 / -0.195 | 0.664 / 1.236 / -0.195 | 0.749 / 1.403 / -0.250 |
| Shrunk ratio + 1.06 floor | 0.680 / 1.314 / -0.161 | 0.630 / 1.152 / -0.191 | 0.660 / 1.194 / -0.137 | 0.692 / 1.379 / -0.166 | 0.671 / 1.253 / -0.152 | 0.752 / 1.400 / -0.237 |
| Fixed ×1.10 | 0.687 / 1.327 / -0.130 | 0.635 / 1.163 / -0.160 | 0.668 / 1.208 / -0.105 | 0.699 / 1.391 / -0.135 | 0.678 / 1.268 / -0.120 | 0.757 / 1.409 / -0.209 |
| 50/50 direct + recursive | 0.671 / 1.293 / -0.190 | 0.614 / 1.121 / -0.226 | 0.643 / 1.167 / -0.180 | 0.687 / 1.362 / -0.190 | 0.663 / 1.233 / -0.181 | 0.742 / 1.376 / -0.266 |
| Recursive | 0.685 / 1.311 / -0.171 | 0.607 / 1.112 / -0.215 | 0.644 / 1.169 / -0.174 | 0.708 / 1.389 / -0.165 | 0.677 / 1.260 / -0.161 | 0.746 / 1.383 / -0.251 |
| Weekday mean | 0.732 / 1.387 / -0.201 | 0.647 / 1.209 / -0.169 | 0.698 / 1.244 / -0.154 | 0.754 / 1.462 / -0.222 | 0.724 / 1.308 / -0.195 | 0.799 / 1.497 / -0.254 |
| 28-day moving average | 0.738 / 1.384 / -0.202 | 0.658 / 1.215 / -0.169 | 0.703 / 1.247 / -0.153 | 0.759 / 1.456 / -0.223 | 0.732 / 1.317 / -0.196 | 0.791 / 1.477 / -0.255 |
| Seasonal naive | 0.828 / 1.587 / -0.261 | 0.737 / 1.419 / -0.231 | 0.808 / 1.473 / -0.218 | 0.845 / 1.650 / -0.279 | 0.818 / 1.530 / -0.261 | 0.906 / 1.667 / -0.261 |
| Production XGBoost | 0.713 / 1.390 / -0.255 | 0.693 / 1.287 / -0.213 | 0.697 / 1.293 / -0.207 | 0.721 / 1.439 / -0.276 | 0.699 / 1.289 / -0.243 | 0.832 / 1.531 / -0.352 |

## Cutoff stability and factor behavior

| Source | Outer cutoff | Inner eligible | Outer eligible | Direct bias | Raw ratio bias | Shrunk bias | Fixed ×1.10 bias | Blend bias |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| real | 2024-09-17 | 36 | 64 | -27.3% | +1.8% | -16.7% | -20.0% | -19.5% |
| real | 2024-10-17 | 49 | 74 | -9.3% | -14.7% | -3.9% | -0.3% | -3.9% |
| real | 2024-11-16 | 61 | 93 | -8.1% | +12.2% | +1.5% | +1.0% | -4.1% |
| real | 2024-12-16 | 64 | 111 | -0.5% | +37.1% | +13.1% | +9.5% | +2.6% |
| synthetic | 2025-07-04 | 252 | 284 | -6.9% | -8.5% | -1.3% | +2.4% | -4.8% |
| synthetic | 2025-08-03 | 262 | 285 | -6.4% | -8.0% | -0.7% | +2.9% | -5.8% |
| synthetic | 2025-09-02 | 271 | 297 | -6.0% | -6.5% | -0.4% | +3.4% | -3.5% |
| synthetic | 2025-10-02 | 284 | 300 | -19.1% | -13.1% | -14.3% | -11.0% | -16.0% |

The eligibility fix reduced the raw ratio’s real overall bias from **+18.6%** in the earlier, survivor-selected run to **+9.9%**. It did not make raw calibration reliable: real cutoff bias ranges from −14.7% to +37.1%, and synthetic overall bias remains −9.2%. The inner and outer eligible counts differ substantially on real data.

- **real:** the 1.06 floor binds in 12 of 24 block/tier cells; median raw factor 1.162, median applied factor 1.061.
- **synthetic:** the 1.06 floor binds in 22 of 24 block/tier cells; median raw factor 1.019, median applied factor 1.060.

The synthetic 20 Nov–31 Dec window remains under-forecast by about 22% for the shrunk variant. Its synthetic overall WAPE (0.787) is effectively tied with recursive (0.787), while MASE rose from 1.289 to 1.314 compared with uncalibrated direct. The fixed ×1.10 control reaches near-zero overall bias but loses additional synthetic WAPE (0.797). The 50/50 blend needs no validation factor and yields 0.759 real / 0.773 synthetic WAPE, with −5.9% / −7.9% bias.

## Decision

Keep **`lgbm_direct` uncalibrated** as the backtesting default. The raw corrected calibration does not meet ±5% observed-sales bias on both sources, and the variants that do rely on constants selected after looking at these same outer folds. No challenger is ready for promotion based on these scores. A prospective shadow run beside `forecast_future_sales_direct` can compare the uncalibrated direct and predeclared 50/50 blend, tracking WAPE, MASE and bias by cutoff, horizon, volume tier, and the Black Friday–Christmas window. Production behavior remains unchanged.

Reproduce: from `src/machine_learning`, run `python backtesting/run_backtest.py --source both --true-demand --models "baselines,current_xgboost,lgbm_direct,lgbm_direct_cal_block_tier,lgbm_direct_cal_shrunk,lgbm_direct_constant_1p10,lgbm_blend_50_50,lgbm_recursive"`. The full run finished in about 11 minutes.
