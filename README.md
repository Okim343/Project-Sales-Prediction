# Sales Forecasting for Mercado Livre

## Project Overview

Sales forecasting application for Mercado Livre using XGBoost models with Multi Step
Direct Forecasting methodology. Models predict 90-day sales forecasts for MLBs (Mercado
Livre identifiers) with continuous learning capabilities.

## Environment Setup

The project uses conda/mamba for environment management:

```bash
mamba env create -f environment.yml
conda activate fcast_project
pre-commit install
```

## Main Scripts

### 1. `script_final.py` - Production Pipeline

Automated production forecasting that generates 90-day forecasts and saves to database.

- Imports fresh data from PostgreSQL
- Processes data and creates time series features
- Generates forecasts for all MLBs
- Saves to `public.mlb_forecasts_90_days` table

**Run**: `python src/machine_learning/script_final.py`

### 2. `script_webapp.py` - Web Dashboard

Web interface for viewing forecasts at http://127.0.0.1:8050

- Interactive forecast charts and data tables
- Filter by MLB, SKU, and date range

**Run**: `python src/web_app/script_webapp.py`

### 3. `pipeline_runner.py` - Unified Continuous Learning Pipeline

Comprehensive pipeline with multiple execution modes:

- **Daily mode**: Incremental updates since last run with validation/rollback
- **Full mode**: Complete training from scratch with fallback capability
- **Since-date mode**: Incremental updates since specified date with data merging
- **Monthly mode**: 6-month sliding window retraining with model comparison

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

## Configuration

### Environment Variables

```bash
export FORECAST_DAYS_LONG=120  # Change forecast duration (default: 90)
export DB_HOST=your_host
export DB_PASSWORD=your_password
export DB_USER=your_username
export DB_NAME=your_database
```

### Configuration Files

- `src/machine_learning/config.py` - Main app configuration
- `src/machine_learning/estimation/model.py` - XGBoost parameters

## Data Flow

1. **Data Import**: PostgreSQL (`public.view_enrico`) → Raw sales data
1. **Processing**: Data cleaning → Time series features → MLB-grouped data
1. **Training**: XGBoost models per MLB with continuous learning
1. **Forecasting**: Multi-step direct 90-day forecasts
1. **Storage**: Database (`public.mlb_forecasts_90_days`) + local files
1. **Visualization**: Web dashboard with interactive charts

## Key Files

- **Config**: `src/machine_learning/config.py`
- **Models**: `src/machine_learning/bld/mlb_regressors.pkl`
- **Data**: `src/machine_learning/data/raw_sql.csv`
- **Pipeline**: `src/machine_learning/pipeline/pipeline_runner.py`
- **Validation**: `src/machine_learning/validation/` (model comparison, validation)

## Database

- **Host**: 172.27.40.210:5432
- **Database**: "Mercado Livre"
- **Input**: `public.view_enrico`
- **Output**: `public.mlb_forecasts_90_days`

## Quick Start

1. **Setup**:

   ```bash
   mamba env create -f environment.yml
   conda activate fcast_project
   export DB_PASSWORD=your_password
   ```

1. **Run**:

   ```bash
   # Production forecasting
   python src/machine_learning/script_final.py

   # Web dashboard
   python src/web_app/script_webapp.py

   # Daily updates (continuous learning)
   python src/machine_learning/pipeline/pipeline_runner.py
   ```

## Architecture

**MLB-Centric Design**: Uses MLB (Mercado Livre identifiers) as primary forecasting
unit.

- **Training**: XGBoost models per MLB
- **Forecasting**: 90-day Multi Step Direct methodology
- **Continuous Learning**: Incremental model updates with validation
- **Data Pipeline**: Metadata tracking with automated rollback capabilities

## Features

- **Continuous Learning**: Daily incremental model updates with performance validation
- **Model Comparison**: Systematic version comparison for monthly retraining decisions
- **Robust Pipeline**: Metadata tracking, automated backups, and rollback mechanisms
- **Validation System**: Integrated model and forecast validation with fallback handling
- **Production Ready**: Comprehensive error handling and logging
- **Web Interface**: Interactive dashboard with real-time data visualization
- **Scalable Architecture**: Batch processing with memory management

## Future Fixes

- **Optimize learning time**: Threading in python and other methods
- **Fix model saving strategy to be able to better read it**: Find a way of saving that
  allows massive models to be read easily
- **Remove useless scripts for final project**: Such as script_final.py or others
