# Direct LightGBM bias-correction backtest

Four 90-day rolling-origin cutoffs per source. Common set: up to the 100 highest-volume production-eligible series per cutoff. Each cell is **WAPE / MASE / bias**; bias is `(forecast − actual) / actual`. The real export covers Dec 2023–Mar 2025; synthetic seed 42 uses the corrected Black Friday calendar. All scores are on observed sales except the true-demand table. True-demand MASE uses the harness's observed-sales seasonal-naive scale as its denominator.

The final direct model validates at `c − 90`, forecasts through `c`, estimates actual/forecast ratios by horizon block and origin-level volume tier, clips each raw ratio to [0.8, 1.5], applies 35% of its move from 1, and floors the applied factor at 1.06. Cells with fewer than 50 predicted validation units use ratio 1 before shrinkage. It then refits through `c`. No order after `c` enters training, validation, features, or factor estimation. The floor and shrinkage were selected after examining these same outer folds; these scores are therefore optimistic for that design choice.

## Full score tables

### Real observed sales

| Variant | Overall | Days 1–7 | Days 8–30 | Days 31+ | Top 20% | Rest |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| direct: shrunk block + tier | 0.769 / 1.131 / +0.010 | 0.520 / 0.877 / +0.020 | 0.715 / 1.175 / +0.030 | 0.824 / 1.143 / -0.000 | 0.734 / 1.269 / +0.059 | 0.837 / 1.094 / -0.083 |
| direct: uncalibrated | 0.748 / 1.090 / -0.108 | 0.515 / 0.858 / -0.046 | 0.701 / 1.143 / -0.064 | 0.798 / 1.097 / -0.135 | 0.713 / 1.230 / -0.061 | 0.815 / 1.053 / -0.200 |
| raw overall ratio | 0.858 / 1.254 / +0.243 | 0.623 / 1.065 / +0.319 | 0.840 / 1.336 / +0.307 | 0.894 / 1.245 / +0.207 | 0.829 / 1.444 / +0.306 | 0.912 / 1.204 / +0.124 |
| raw block ratios | 0.833 / 1.221 / +0.185 | 0.524 / 0.894 / +0.029 | 0.763 / 1.237 / +0.125 | 0.901 / 1.253 / +0.229 | 0.803 / 1.399 / +0.248 | 0.889 / 1.174 / +0.065 |
| raw tier ratios | 0.838 / 1.262 / +0.210 | 0.615 / 1.091 / +0.285 | 0.813 / 1.339 / +0.269 | 0.876 / 1.253 / +0.176 | 0.794 / 1.394 / +0.237 | 0.920 / 1.228 / +0.160 |
| raw block + tier ratios | 0.828 / 1.237 / +0.186 | 0.524 / 0.883 / +0.038 | 0.751 / 1.268 / +0.123 | 0.899 / 1.266 / +0.231 | 0.790 / 1.382 / +0.223 | 0.901 / 1.198 / +0.115 |
| inner-selected Tweedie power | 0.763 / 1.131 / -0.065 | 0.527 / 0.888 / -0.045 | 0.717 / 1.183 / -0.044 | 0.812 / 1.139 / -0.077 | 0.728 / 1.261 / -0.036 | 0.829 / 1.096 / -0.121 |
| level target / 28-day mean | 0.771 / 1.131 / -0.088 | 0.521 / 0.864 / -0.059 | 0.715 / 1.170 / -0.064 | 0.825 / 1.146 / -0.102 | 0.727 / 1.256 / -0.115 | 0.853 / 1.097 / -0.036 |
| level target / 91-day mean | 0.771 / 1.144 / -0.055 | 0.526 / 0.865 / -0.080 | 0.708 / 1.173 / -0.057 | 0.829 / 1.165 / -0.051 | 0.727 / 1.268 / -0.090 | 0.855 / 1.111 / +0.012 |
| trend feature (removed) | 0.750 / 1.089 / -0.100 | 0.514 / 0.854 / -0.047 | 0.703 / 1.142 / -0.058 | 0.800 / 1.096 / -0.125 | 0.718 / 1.246 / -0.050 | 0.811 / 1.047 / -0.195 |
| recursive reference | 0.789 / 1.196 / -0.009 | 0.510 / 0.866 / -0.039 | 0.714 / 1.197 / -0.035 | 0.856 / 1.234 / +0.005 | 0.746 / 1.296 / -0.008 | 0.871 / 1.169 / -0.011 |
| weekday-mean baseline | 0.851 / 1.254 / +0.121 | 0.596 / 0.998 / +0.105 | 0.763 / 1.255 / +0.032 | 0.920 / 1.284 / +0.162 | 0.826 / 1.409 / +0.155 | 0.897 / 1.213 / +0.055 |
| production XGBoost | 0.908 / 1.435 / +0.081 | 0.731 / 1.249 / -0.065 | 0.808 / 1.415 / -0.071 | 0.973 / 1.464 / +0.165 | 0.823 / 1.463 / +0.107 | 1.070 / 1.427 / +0.032 |

### Synthetic observed sales

| Variant | Overall | Days 1–7 | Days 8–30 | Days 31+ | Top 20% | Rest |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| direct: shrunk block + tier | 0.787 / 1.314 / -0.046 | 0.646 / 1.072 / -0.072 | 0.793 / 1.207 / +0.001 | 0.800 / 1.384 / -0.060 | 0.779 / 1.280 / -0.039 | 0.849 / 1.363 / -0.109 |
| direct: uncalibrated | 0.773 / 1.289 / -0.101 | 0.637 / 1.054 / -0.125 | 0.776 / 1.181 / -0.055 | 0.786 / 1.358 / -0.113 | 0.765 / 1.252 / -0.093 | 0.837 / 1.341 / -0.161 |
| raw overall ratio | 0.777 / 1.297 / -0.080 | 0.643 / 1.063 / -0.104 | 0.783 / 1.189 / -0.035 | 0.789 / 1.366 / -0.093 | 0.770 / 1.261 / -0.073 | 0.841 / 1.347 / -0.142 |
| raw block ratios | 0.777 / 1.298 / -0.079 | 0.636 / 1.051 / -0.150 | 0.773 / 1.175 / -0.068 | 0.794 / 1.374 / -0.076 | 0.770 / 1.263 / -0.072 | 0.841 / 1.348 / -0.142 |
| raw tier ratios | 0.776 / 1.304 / -0.083 | 0.644 / 1.066 / -0.108 | 0.784 / 1.197 / -0.039 | 0.788 / 1.373 / -0.096 | 0.768 / 1.261 / -0.079 | 0.847 / 1.365 / -0.118 |
| raw block + tier ratios | 0.776 / 1.306 / -0.083 | 0.636 / 1.054 / -0.153 | 0.773 / 1.179 / -0.070 | 0.793 / 1.384 / -0.079 | 0.768 / 1.262 / -0.078 | 0.847 / 1.367 / -0.118 |
| inner-selected Tweedie power | 0.773 / 1.291 / -0.093 | 0.642 / 1.060 / -0.126 | 0.778 / 1.182 / -0.050 | 0.786 / 1.360 / -0.104 | 0.766 / 1.255 / -0.088 | 0.838 / 1.341 / -0.138 |
| level target / 28-day mean | 0.796 / 1.316 / -0.096 | 0.647 / 1.063 / -0.130 | 0.797 / 1.189 / -0.048 | 0.811 / 1.395 / -0.109 | 0.789 / 1.286 / -0.097 | 0.854 / 1.358 / -0.093 |
| level target / 91-day mean | 0.806 / 1.352 / -0.078 | 0.657 / 1.077 / -0.132 | 0.801 / 1.213 / -0.046 | 0.824 / 1.438 / -0.083 | 0.798 / 1.323 / -0.082 | 0.874 / 1.393 / -0.043 |
| trend feature (removed) | 0.769 / 1.282 / -0.100 | 0.636 / 1.055 / -0.125 | 0.773 / 1.177 / -0.045 | 0.782 / 1.348 / -0.116 | 0.761 / 1.242 / -0.092 | 0.835 / 1.337 / -0.163 |
| recursive reference | 0.787 / 1.310 / -0.058 | 0.636 / 1.052 / -0.100 | 0.768 / 1.178 / -0.042 | 0.810 / 1.390 / -0.058 | 0.781 / 1.285 / -0.050 | 0.841 / 1.344 / -0.126 |
| weekday-mean baseline | 0.838 / 1.377 / -0.092 | 0.735 / 1.200 / -0.047 | 0.832 / 1.251 / -0.018 | 0.851 / 1.446 / -0.122 | 0.831 / 1.323 / -0.088 | 0.898 / 1.452 / -0.130 |
| production XGBoost | 0.814 / 1.353 / -0.153 | 0.800 / 1.256 / -0.097 | 0.822 / 1.264 / -0.079 | 0.813 / 1.398 / -0.184 | 0.801 / 1.286 / -0.142 | 0.919 / 1.447 / -0.243 |

### Synthetic planted true demand

| Variant | Overall | Days 1–7 | Days 8–30 | Days 31+ | Top 20% | Rest |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| direct: shrunk block + tier | 0.680 / 1.314 / -0.161 | 0.630 / 1.152 / -0.191 | 0.660 / 1.194 / -0.137 | 0.692 / 1.379 / -0.166 | 0.671 / 1.253 / -0.152 | 0.752 / 1.400 / -0.237 |
| direct: uncalibrated | 0.672 / 1.298 / -0.209 | 0.625 / 1.139 / -0.237 | 0.652 / 1.177 / -0.186 | 0.684 / 1.362 / -0.214 | 0.663 / 1.234 / -0.200 | 0.746 / 1.387 / -0.281 |
| raw overall ratio | 0.675 / 1.303 / -0.191 | 0.630 / 1.145 / -0.219 | 0.656 / 1.182 / -0.168 | 0.686 / 1.367 / -0.196 | 0.666 / 1.240 / -0.182 | 0.747 / 1.390 / -0.265 |
| raw block ratios | 0.675 / 1.304 / -0.190 | 0.627 / 1.139 / -0.258 | 0.651 / 1.174 / -0.197 | 0.689 / 1.372 / -0.180 | 0.666 / 1.241 / -0.181 | 0.747 / 1.390 / -0.265 |
| raw tier ratios | 0.674 / 1.307 / -0.194 | 0.630 / 1.148 / -0.222 | 0.657 / 1.189 / -0.172 | 0.685 / 1.371 / -0.198 | 0.665 / 1.239 / -0.188 | 0.750 / 1.402 / -0.244 |
| raw block + tier ratios | 0.675 / 1.308 / -0.193 | 0.627 / 1.141 / -0.262 | 0.652 / 1.176 / -0.199 | 0.688 / 1.378 / -0.184 | 0.665 / 1.240 / -0.187 | 0.750 / 1.404 / -0.244 |
| inner-selected Tweedie power | 0.672 / 1.296 / -0.202 | 0.630 / 1.142 / -0.238 | 0.654 / 1.178 / -0.181 | 0.683 / 1.359 / -0.206 | 0.664 / 1.236 / -0.195 | 0.741 / 1.380 / -0.261 |
| level target / 28-day mean | 0.696 / 1.323 / -0.205 | 0.639 / 1.149 / -0.241 | 0.672 / 1.186 / -0.179 | 0.710 / 1.395 / -0.210 | 0.688 / 1.269 / -0.203 | 0.757 / 1.397 / -0.223 |
| level target / 91-day mean | 0.704 / 1.354 / -0.189 | 0.644 / 1.156 / -0.243 | 0.678 / 1.213 / -0.178 | 0.719 / 1.431 / -0.187 | 0.696 / 1.301 / -0.190 | 0.771 / 1.427 / -0.180 |
| trend feature (removed) | 0.668 / 1.289 / -0.208 | 0.622 / 1.136 / -0.236 | 0.645 / 1.169 / -0.177 | 0.681 / 1.353 / -0.216 | 0.658 / 1.222 / -0.199 | 0.744 / 1.383 / -0.283 |
| recursive reference | 0.685 / 1.311 / -0.171 | 0.607 / 1.112 / -0.215 | 0.644 / 1.169 / -0.174 | 0.708 / 1.389 / -0.165 | 0.677 / 1.260 / -0.161 | 0.746 / 1.383 / -0.251 |
| weekday-mean baseline | 0.732 / 1.387 / -0.201 | 0.647 / 1.209 / -0.169 | 0.698 / 1.244 / -0.154 | 0.754 / 1.462 / -0.222 | 0.724 / 1.308 / -0.195 | 0.799 / 1.497 / -0.254 |
| production XGBoost | 0.713 / 1.390 / -0.255 | 0.693 / 1.287 / -0.213 | 0.697 / 1.293 / -0.207 | 0.721 / 1.439 / -0.276 | 0.699 / 1.289 / -0.243 | 0.832 / 1.531 / -0.352 |

## Interpretation

- The calibrated default reaches +1.0% overall observed-sales bias on real data and −4.6% on synthetic data. WAPE rises by 0.021 and 0.014 versus the uncalibrated direct model, while remaining below the weekday-mean baseline on both sources.
- The rest tier remains under-forecast: −8.3% real and −10.9% synthetic. The synthetic 20 Nov–31 Dec window is −21.7%; the 2 Oct cutoff alone is −14.3%, while the first three synthetic cutoffs are within about 1.3% of zero. Holiday transfer and long-horizon growth remain the main risks.
- Planted true-demand bias is −16.1% for the recommended model. Stockout censoring and growth make this a different target; the model is calibrated to observed sales.
- Raw validation ratios overcorrected on real data (+18% to +24% bias) yet barely corrected synthetic data (about −8% bias). Inner power selection and level-normalised targets did not meet the joint bias and accuracy goal. The trend feature changed bias little, so it was removed from the code.
- The complete final command finished in about 11 minutes. Reproduce with `cd src/machine_learning && python backtesting/run_backtest.py --source both --true-demand --models "baselines,current_xgboost,lgbm_direct,lgbm_direct_uncalibrated,lgbm_recursive"`. Raw variants are selectable through `LGBM_VARIANTS`; run with `--skip-current` for faster ablations.

## Next step

Run the calibrated challenger in shadow mode beside `forecast_future_sales_direct`, without publishing its forecasts. Use future cutoffs that were not used to choose the 0.35 shrinkage and 1.06 floor. Check observed-sales bias and WAPE by horizon, tier, and the Black Friday–Christmas window before considering any production change.
