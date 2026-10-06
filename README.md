<div align="center">

# 📈 Project Sales Prediction

**Per-listing sales forecasting for Mercado Livre, with continuous learning, automated
validation, and a live dashboard.**

![Python](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/model-XGBoost-EB5E28)
![PostgreSQL](https://img.shields.io/badge/data-PostgreSQL-4169E1?logo=postgresql&logoColor=white)
![Dash](https://img.shields.io/badge/dashboard-Plotly%20Dash-3F4F75?logo=plotly&logoColor=white)
![License](https://img.shields.io/badge/license-Proprietary-red)

[Overview](#-overview) • [Features](#-features) • [Architecture](#-architecture) •
[Getting started](#-getting-started) • [Usage](#-usage) •
[Configuration](#%EF%B8%8F-configuration) • [License](#-license)

</div>

______________________________________________________________________

## 🔎 Overview

Project Sales Prediction trains an individual **XGBoost regressor for every active
Mercado Livre listing (MLB)** and produces a **90-day daily sales forecast** for each
one.

Sales history is read from PostgreSQL, cleaned, and turned into time-series features
(calendar effects, lags, rolling means, price). Forecasts are written back to the
database, where they feed downstream planning and an interactive Dash dashboard.

The system is built to run unattended. It updates models incrementally as new data
arrives, validates every model and forecast before publishing, rolls back automatically
when quality degrades, and keeps a full audit trail of every run.

## ✨ Features

|                              |                                                                                                                       |
| ---------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| 🎯 **Per-listing models**    | One XGBoost model per MLB, capturing listing-specific demand patterns                                                 |
| 🔭 **90-day horizon**        | Direct multi-step forecasting; horizon configurable via environment variable                                          |
| 🔁 **Continuous learning**   | Daily incremental updates, full retrains, date-bounded backfills, and monthly sliding-window refreshes                |
| 🛡️ **Validation & rollback** | Integrated model and forecast checks; models are archived and restored automatically if performance drops             |
| 🧾 **Run metadata**          | Every run and rollback decision is logged to `public.pipeline_metadata`                                               |
| 🚀 **Deployment-ready**      | Environment validation, rotating logs, cleanup handling, and cron launchers with meaningful exit codes                |
| 📊 **Dashboard**             | Dash + Plotly UI with MLB/SKU/date filters, 15-minute caching, and graceful fallback when the database is unavailable |
| 🧪 **Dry-run mirrors**       | Every entry point has a `test_*` twin limited to a handful of MLBs for fast, safe checks                              |

## 🏗 Architecture

```mermaid
flowchart LR
    A[(PostgreSQL<br/>sales view)] --> B[Import &<br/>cleaning]
    B --> C[Feature<br/>engineering]
    C --> D[Per-MLB XGBoost<br/>training / update]
    D --> E{Integrated<br/>validation}
    E -- pass --> F[90-day<br/>forecasts]
    E -- fail --> G[Rollback to<br/>archived models]
    G --> F
    F --> H[(Forecast table)]
    F --> I[(Model store<br/>bld/*.pkl)]
    H --> J[Dash dashboard]
    D -. run metadata .-> K[(pipeline_metadata)]
```

### Pipeline modes

| Mode                | What it does                                                                      | Typical schedule |
| ------------------- | --------------------------------------------------------------------------------- | ---------------- |
| `daily` *(default)* | Continues training existing models on data since the last successful run          | Every day        |
| `monthly`           | Retrains on a six-month sliding window and keeps the better of old vs. new models | Monthly          |
| `full`              | Clean retrain on all history, with automatic fallback to archived models          | On demand        |
| `--since-date`      | Rebuilds models using data from a given date onward                               | Backfills        |

### Project structure

```text
Project-Sales-Prediction/
├── environment.yml                 # Conda environment (fcast_project)
├── LICENSE
└── src/
    ├── machine_learning/
    │   ├── config.py               # Central configuration (env-driven)
    │   ├── script_final.py         # Full production refresh
    │   ├── deploy_pipeline.py      # Hardened deployment entry point
    │   ├── pipeline/               # Unified runner + integrated validator
    │   ├── data_management/        # SQL import, cleaning, features, metadata
    │   ├── estimation/             # Training, forecasting, model storage
    │   ├── validation/             # Model/forecast validators, model comparison
    │   ├── deployment/             # Env checks, logging, paths, cleanup
    │   ├── cron_scripts/           # Cron launchers (prod + test)
    │   ├── backtesting/            # Synthetic data generator + rolling-origin backtest
    │   └── notebooks/              # Exploratory analysis
    └── web_app/
        └── script_webapp.py        # Dash dashboard
```

## 🚀 Getting started

### Prerequisites

- [Mamba](https://mamba.readthedocs.io/) or Conda
- Network access to the PostgreSQL instance holding the sales data

### Installation

```bash
git clone https://github.com/Okim343/Project-Sales-Prediction.git
cd Project-Sales-Prediction
mamba env create -f environment.yml
conda activate fcast_project
pre-commit install
```

Set your database credentials (or put them in a `.env` file at the project root):

```bash
export DB_HOST=your_host
export DB_USER=your_user
export DB_PASSWORD=your_password
export DB_NAME=your_database
```

## 🧭 Usage

**Continuous learning pipeline**

```bash
python src/machine_learning/pipeline/pipeline_runner.py                       # daily
python src/machine_learning/pipeline/pipeline_runner.py --mode=monthly
python src/machine_learning/pipeline/pipeline_runner.py --mode=full
python src/machine_learning/pipeline/pipeline_runner.py --since-date=2025-01-15
```

**Production deployment** (environment checks, rotating logs, safe cleanup)

```bash
python src/machine_learning/deploy_pipeline.py --mode=daily
```

**Scheduling with cron**

```cron
0 3 * * *  bash /path/to/Project-Sales-Prediction/src/machine_learning/cron_scripts/daily_cron.sh
0 4 1 * *  bash /path/to/Project-Sales-Prediction/src/machine_learning/cron_scripts/monthly_cron.sh
```

**Full production refresh**

```bash
python src/machine_learning/script_final.py
```

**Dashboard** at <http://127.0.0.1:8050>

```bash
python src/web_app/script_webapp.py
```

> [!TIP]
> Each entry point has a `test_*` counterpart (for example
> `pipeline/test_pipeline_runner.py` or `cron_scripts/test_daily_cron.sh`) that runs the
> same logic on a small subset of MLBs and writes to a separate test table.

## 📏 Backtesting

`src/machine_learning/backtesting/` scores the production model against simple baselines
with a rolling-origin backtest. For each cutoff, every model sees only data up to that
date and is scored on the next 90 days. Results are averaged over 4 cutoffs, 30 days
apart.

It runs on two sources:

- **Historical export**: `data/raw_sql.csv` (git-ignored; SKU-level, Dec 2023 – Mar
  2025).
- **Synthetic data**: generated to match the production view's columns and calibrated to
  the historical export. It has weekday pattern, growth, yearly seasonality, Black
  Friday and Christmas, price discounts, stockouts, and staggered launches. The planted
  parameters and true pre-stockout demand are saved as ground truth.

```bash
cd src/machine_learning
python backtesting/synthetic_data.py                        # (re)generate data/synthetic_orders.csv
python backtesting/run_backtest.py --source both            # full run incl. LightGBM, about 10 minutes
python backtesting/run_backtest.py --source real --skip-current   # baselines only, seconds
python -m pytest backtesting -q
```

Results land in `bld/backtest/<source>_<timestamp>/` (`summary.md`, `scores.csv`,
`forecasts.parquet`).

| Metric | Meaning                                                                                             |
| ------ | --------------------------------------------------------------------------------------------------- |
| WAPE   | Total absolute error / total units sold. The headline number; lower is better.                      |
| MASE   | Error relative to a 7-day seasonal naive forecast on the training data; below 1 beats it in-sample. |
| Bias   | Total error / total units sold. Positive means over-forecasting.                                    |

Baselines: `seasonal_naive_7` (repeat last week), `weekday_mean_4w` (mean of the same
weekday over the last 4 weeks), `moving_average_28` (flat 28-day mean). A new model
should beat all three before it replaces the production one.

### LightGBM challenger

`backtesting/lightgbm_model.py` trains **global LightGBM regressors** (Tweedie
objective) across series up to the cutoff, instead of one XGBoost per listing. Features
describe the target date (weekday, day of month, Black Friday, Christmas, Brazilian
holidays) and each series' recent sales (lags, rolling means, same-weekday mean), last
known unit price, volume level and identity. Two strategies are available: `direct` (one
model per horizon block, days 1–7 / 8–30 / 31–90, using only values known at the origin)
and `recursive` (one-step model fed its own predictions). A stockout proxy drops
implausible zero-sales runs from the training targets.

```bash
python backtesting/run_backtest.py --models "baselines,lgbm_direct"               # challenger only
python backtesting/run_backtest.py --models "baselines,lgbm_direct*" --skip-current  # ablations
python backtesting/run_backtest.py --source synthetic --true-demand                # also score vs true demand
```

`--models` takes names or glob patterns; the variants are listed in `LGBM_VARIANTS`. The
default `lgbm_direct` now calibrates on an inner 90-day window ending at each cutoff. It
estimates an observed-sales ratio for each horizon block and origin-level volume tier,
clips ratios to \[0.8, 1.5\], then applies 35% of the adjustment with a 1.06 minimum
factor. The previous model remains available as `lgbm_direct_uncalibrated`. All
validation, factors and features use data available by the cutoff. The shrinkage and
floor were selected on these same backtest folds, so an independent shadow run is needed
before production use.

Common-set WAPE (bias), 4 cutoffs, observed sales:

| model                      | real              | synthetic         |
| -------------------------- | ----------------- | ----------------- |
| `lgbm_direct`              | **0.769** (+0.01) | **0.787** (−0.05) |
| `lgbm_direct_uncalibrated` | 0.748 (−0.11)     | 0.773 (−0.10)     |
| `lgbm_recursive`           | 0.789 (−0.01)     | 0.787 (−0.06)     |
| `weekday_mean_4w`          | 0.851 (+0.12)     | 0.838 (−0.09)     |
| `current_xgboost`          | 0.908 (+0.08)     | 0.814 (−0.15)     |

The calibrated model keeps the lead over `weekday_mean_4w` on both sources, with overall
bias inside ±5%. Lower-volume series remain under-forecast (−8% real, −11% synthetic),
and the synthetic Black Friday–Christmas window remains difficult. See
[`BIAS_CORRECTION_RESULTS.md`](src/machine_learning/backtesting/BIAS_CORRECTION_RESULTS.md)
for all variant, horizon, volume-tier and true-demand scores.

## ⚙️ Configuration

All settings live in [`src/machine_learning/config.py`](src/machine_learning/config.py)
and can be overridden through environment variables.

| Variable                    | Default                        | Description                                           |
| --------------------------- | ------------------------------ | ----------------------------------------------------- |
| `DB_HOST` / `DB_PORT`       | — / `5432`                     | PostgreSQL host and port                              |
| `DB_USER` / `DB_PASSWORD`   | —                              | Database credentials                                  |
| `DB_NAME`                   | `Mercado Livre`                | Database name                                         |
| `DB_VIEW`                   | `public.view_enrico`           | Input sales view                                      |
| `DB_FORECAST_TABLE`         | `public.mlb_forecasts_90_days` | Forecast output table                                 |
| `FORECAST_DAYS_LONG`        | `90`                           | Production forecast horizon (days)                    |
| `FORECAST_DAYS`             | `30`                           | Short horizon used by legacy/test helpers             |
| `ACTIVE_MLB_DAYS_THRESHOLD` | `30`                           | Days of recent activity for an MLB to count as active |
| `TEST_MLB_COUNT`            | `5`                            | Number of MLBs used by `test_*` scripts               |

XGBoost hyperparameters are defined in
[`src/machine_learning/estimation/model.py`](src/machine_learning/estimation/model.py).

## 🗺 Roadmap

- [ ] Alerting for data freshness, pipeline failures, and forecast anomalies
- [ ] Automated feature-drift tracking for incremental runs
- [ ] Faster, leaner incremental training for large MLB sets

## 📄 License

**Proprietary — all rights reserved.** © 2025–2026 Enrico Truzzi.

This repository is not open source. Viewing the code does not grant permission to run,
copy, modify, or distribute it. Any use requires prior written permission from the
author. See [`LICENSE`](LICENSE) for the full terms.
