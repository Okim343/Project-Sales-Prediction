# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in
this repository.

## Project Overview

This is a sales forecasting application designed for Mercado Livre users. The project
implements machine learning models (XGBoost) to predict sales for different SKUs using
time series data, with Multi Step Direct Forecasting methodology.

## Environment Setup

The project uses conda/mamba for environment management:

```bash
mamba env create -f environment.yml
conda activate fcast_project
pre-commit install
```

## Script Overview

The project has three main execution scripts, each serving different purposes:

### 1. `script_final.py` - Production Forecasting Pipeline

**Purpose**: Automated production script for generating forecasts and saving to
database.

**What it does**:

- Imports fresh data from PostgreSQL database
- Processes and cleans the sales data
- Creates time series features
- Generates 90-day forecasts for all SKUs
- Saves forecasts directly to the database (`public.sku_forecats_90_days` table)
- No user interaction required

**When to use**: For automated/production runs, scheduled forecasting, or when you want
to update the database with fresh forecasts.

**Run with**: `python src/machine_learning/script_final.py`

### 2. `script_main.py` - Interactive Model Training & Predictions

**Purpose**: Interactive script for training models and visualizing individual SKU
predictions.

**What it does**:

- Loads data (cached or fresh from database)
- Processes data and creates features
- Trains XGBoost models for each SKU (or loads existing models)
- Generates sample time series plots
- Allows interactive SKU selection for prediction visualization
- Saves prediction plots as PNG files

**When to use**: For model development, training new models, or analyzing individual SKU
predictions.

**Run with**: `python src/machine_learning/script_main.py`

### 3. `script_main_forecast.py` - Interactive Forecasting

**Purpose**: Interactive script for generating and visualizing forecasts with user
input.

**What it does**:

- Loads data (cached or fresh from database)
- Processes data and creates features
- Generates 30-day forecasts for all SKUs (or loads existing forecasts)
- Allows interactive SKU selection for forecast visualization
- Saves forecast plots as HTML files
- Saves forecasts to database

**When to use**: For interactive forecasting sessions, exploring forecasts for specific
SKUs, or generating forecast visualizations.

**Run with**: `python src/machine_learning/script_main_forecast.py`

### 4. `script_webapp.py` - Web Dashboard

**Purpose**: Web-based dashboard for viewing forecasts.

**What it does**:

- Launches a web interface at http://127.0.0.1:8050
- Displays interactive forecast charts and data tables
- Includes data caching for performance
- Allows filtering by SKU and date range

**Run with**: `python src/web_app/script_webapp.py`

## Configuration & Customization

### Changing Forecast Duration

**Method 1: Environment Variables (Recommended)** Set these before running scripts:

```bash
export FORECAST_DAYS=60        # For script_main_forecast.py (default: 30)
export FORECAST_DAYS_LONG=120  # For script_final.py (default: 90)
```

**Method 2: Edit Configuration File** Modify `src/machine_learning/config.py`:

```python
class AppConfig:
    FORECAST_DAYS = 45  # Change from 30 to desired days
    FORECAST_DAYS_LONG = 120  # Change from 90 to desired days
```

### Other Configuration Options

**Database Settings**:

```bash
export DB_HOST=your_host
export DB_PASSWORD=your_password
export DB_USER=your_username
export DB_NAME=your_database
```

**Model Parameters**: Edit `src/machine_learning/estimation/model.py`:

```python
XGBOOST_PARAMS = {
    "n_estimators": 1000,  # Number of trees
    "max_depth": 3,  # Tree depth
    "learning_rate": 0.01,  # Learning rate
    # ... other parameters
}
```

**Script Behavior**: Edit the boolean flags in each script:

```python
# In script_main.py and script_main_forecast.py
import_data = False  # Set to True to fetch fresh data
import_regressor = True  # Set to False to train new models
```

## Data Flow

1. **Data Import**: Raw sales data from PostgreSQL (`public.view_enrico`)
1. **Data Processing**: Cleaning and time series feature creation
1. **Model Training**: XGBoost regressors trained per SKU
1. **Forecasting**: Multi-step direct forecasting
1. **Storage**: Results saved to database and local files
1. **Visualization**: Web dashboard and plot generation

## Key File Locations

- **Configuration**: `src/machine_learning/config.py` - Main configuration
- **Models**: `src/machine_learning/bld/sku_regressors.pkl`
- **Forecasts**: `src/machine_learning/bld/sku_forecast.pkl`
- **Data**: `src/machine_learning/data/raw_sql.csv`
- **Plots**: `src/machine_learning/bld/*.html` and `*.png`

## Database Configuration

- **Host**: 172.27.40.210:5432
- **Database**: "Mercado Livre"
- **Input View**: `public.view_enrico`
- **Output Table**: `public.sku_forecats_90_days`

## Quick Start Guide

1. **Setup Environment**:

   ```bash
   mamba env create -f environment.yml
   conda activate fcast_project
   ```

1. **Set Database Password** (if needed):

   ```bash
   export DB_PASSWORD=your_password
   ```

1. **Run Scripts**:

   - **For production forecasting**: `python src/machine_learning/script_final.py`
   - **For model training**: `python src/machine_learning/script_main.py`
   - **For interactive forecasting**:
     `python src/machine_learning/script_main_forecast.py`
   - **For web dashboard**: `python src/web_app/script_webapp.py`

## Testing

- Run tests: `pytest`
- Run tests with coverage: `pytest --cov`
- Run tests in parallel: `pytest -n auto`

## Development Notes

- Models are trained per SKU individually using XGBoost
- The system supports both model training and loading pre-trained models
- All scripts include comprehensive error handling and logging
- The web application includes data caching for improved performance
- Forecasts are generated using Multi Step Direct Forecasting methodology
