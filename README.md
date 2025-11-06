# Sales Forecasting for Mercado Livre

## Project Overview

Sales forecasting application for Mercado Livre powered by XGBoost models trained per
MLB (listing). The system delivers 90-day forecasts, supports incremental learning with
automated validation/rollback, and ships with production-ready deployment tooling plus a
Dash dashboard.

## Environment Setup

The project uses conda/mamba for environment management:

```bash
mamba env create -f environment.yml
conda activate fcast_project
pre-commit install
```

## Main Scripts

### 1. `script_final.py` - Production Pipeline

Automated production refresh that trains models from scratch and publishes forecasts.

- Imports the latest data from PostgreSQL
- Cleans data and builds time-series features
- Trains or updates XGBoost models for every active MLB
- Generates 90-day forecasts using direct multi-step methodology
- Saves results to `public.mlb_forecasts_90_days`
- Logs run metadata and persists models to `bld/mlb_regressors.pkl`

**Run**: `python src/machine_learning/script_final.py`

### 2. `pipeline_runner.py` - Continuous Learning Orchestrator

Unified CLI that powers all continuous learning workflows.

- **Daily mode**: Incremental updates with integrated validation and rollback
- **Full mode**: Clean retraining with automatic fallback if performance drops
- **Since-date mode**: Rebuild using data from a specified date onward
- **Monthly mode**: Six-month sliding-window retraining with model comparison
- Archives models, logs metadata, and writes validated forecasts back to SQL

**Run**:

```bash
# Daily mode (default)
python src/machine_learning/pipeline/pipeline_runner.py

# Full mode
python src/machine_learning/pipeline/pipeline_runner.py --mode=full

# Monthly mode
python src/machine_learning/pipeline/pipeline_runner.py --mode=monthly

# Since specific date
python src/machine_learning/pipeline/pipeline_runner.py --since-date=2024-01-15
```

### 3. `deploy_pipeline.py` - Deployment Wrapper

Production-safe wrapper that prepares the environment, then calls the pipeline runner.

- Validates environment variables, paths, and Python interpreter
- Configures production logging and enriched context metadata
- Cleans up working directories and returns meaningful exit codes
- Ideal for cron jobs and VPS deployment

**Run**:

```bash
python src/machine_learning/deploy_pipeline.py --mode=daily
python src/machine_learning/deploy_pipeline.py --mode=full
python src/machine_learning/deploy_pipeline.py --mode=monthly
python src/machine_learning/deploy_pipeline.py --since-date=2024-01-15
```

### 4. `script_webapp.py` - Web Dashboard

Dash-based dashboard available at http://127.0.0.1:8050.

- Cached database reads with 15-minute TTL to cut down load
- Filters by MLB, SKU, and date range with Plotly visualizations
- Graceful fallback to cached data when the database is unavailable
- Includes lightweight session handling scaffolding

**Run**: `python src/web_app/script_webapp.py`

> Tip: For cron scheduling use the wrappers in `src/machine_learning/cron_scripts/`.

## Configuration

### Environment Variables

```bash
export FORECAST_DAYS_LONG=120  # Change forecast duration (default: 90)
export DB_HOST=your_host
export DB_PASSWORD=your_password
export DB_USER=your_username
export DB_NAME=your_database
export DB_VIEW=public.view_enrico
export DB_FORECAST_TABLE=public.mlb_forecasts_90_days
```

### Configuration Files

- `src/machine_learning/config.py` - Main app configuration
- `src/machine_learning/estimation/model.py` - XGBoost parameters
- `src/machine_learning/estimation/model_storage.py` - Model persistence helpers
- `src/machine_learning/data_management/metadata_tracker.py` - Pipeline run logging
- `src/machine_learning/deployment/` - Production logging, environment validation,
  cleanup utilities

## Data Flow

1. **Data Import**: PostgreSQL (`public.view_enrico`) → Raw sales data
1. **Processing**: Cleaning, feature engineering, metadata logging
1. **Training**: XGBoost models per MLB with optional incremental continuation
1. **Validation**: Forecast sanity checks, model comparison, rollback handling
1. **Forecasting**: Multi-step direct 90-day forecasts
1. **Storage**: Database (`public.mlb_forecasts_90_days`) + local serialized models
1. **Visualization**: Dash dashboard and saved artifacts

## Key Files

- **Config**: `src/machine_learning/config.py`
- **Models**: `bld/mlb_regressors.pkl`
- **Forecasts**: `bld/mlb_forecast.pkl`
- **Data**: `data/raw_sql.csv`
- **Pipeline**: `src/machine_learning/pipeline/pipeline_runner.py`
- **Deployment**: `src/machine_learning/deploy_pipeline.py`
- **Validation Suite**: `src/machine_learning/pipeline/integrated_validator.py`,
  `src/machine_learning/validation/`
- **Cron Scripts**: `src/machine_learning/cron_scripts/*.sh`

## Database

- **Host**: 172.27.40.210:5432
- **Database**: "Mercado Livre"
- **Input**: `public.view_enrico`
- **Forecast Output**: `public.mlb_forecasts_90_days` (override via `DB_FORECAST_TABLE`)
- **Metadata Table**: `public.pipeline_metadata` (auto-managed)

## Quick Start

1. **Setup**:

   ```bash
   mamba env create -f environment.yml
   conda activate fcast_project
   export DB_PASSWORD=your_password
   ```

1. **Run**:

   ```bash
   # Production full refresh
   python src/machine_learning/script_final.py

   # Daily continuous learning
   python src/machine_learning/pipeline/pipeline_runner.py

   # Production-safe execution (cron/VPS)
   python src/machine_learning/deploy_pipeline.py --mode=daily

   # Web dashboard
   python src/web_app/script_webapp.py
   ```

## Architecture

**MLB-Centric Design**: Each MLB gets its own XGBoost regressor backed by direct
multi-step forecasting.

- **Incremental Learning**: Daily mode continues training existing models with
  validation + rollback
- **Monthly Refresh**: Six-month sliding window retraining with model comparison
- **Integrated Validation**: Forecast and model checks guard against regressions
- **Metadata & Archiving**: Every run logged; models archived before risky updates
- **Deployment Wrapper**: Dedicated tooling separates ML logic from ops concerns

## Features

- **Integrated Validation & Rollback**: Automated checks on models and forecasts with
  safe fallback
- **Incremental Model Updates**: Daily mode supports continuation training with
  statistical guards
- **Model Archiving & Metadata**: Persistent run history plus timestamped model backups
- **Deployment-Ready Execution**: Environment validation, logging, cron scripts, and
  graceful cleanup
- **Dash Dashboard**: Cached data access, MLB/SKU filters, Plotly visuals, login
  scaffolding
- **Robust Logging**: Consistent logging across pipeline, deployment, and web layers
- **Configurable Horizons**: Forecast horizon adjustable through environment variables
  or config

## Future Fixes

- **Production Monitoring**: Add alerting for data freshness, pipeline failures, and
  forecast anomalies
- **Data Drift Tracking**: Automate feature drift metrics and dashboards for incremental
  runs
- **Performance Tuning**: Further optimize incremental training time and memory usage
  for large MLB sets
