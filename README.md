# README.md

## Project Overview

This is a sales forecasting application designed for Mercado Livre users. The project
implements machine learning models (XGBoost) to predict sales for different MLBs
(Mercado Livre identifiers) using time series data, with Multi Step Direct Forecasting
methodology. Each MLB serves as the unique identifier for forecasting, while SKU
information is preserved for reference. Multiple SKUs can share the same MLB, but each
MLB maps to exactly one SKU.

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

- Imports fresh data from PostgreSQL database (using direct import for data
  completeness)
- Validates data freshness to ensure recent data is available
- Processes and cleans the sales data with improved timezone handling
- Creates time series features
- Generates 90-day forecasts for all MLBs using normal rounding (to nearest integer)
- Saves forecasts directly to the database (`public.mlb_forecasts_90_days` table) with
  both MLB and SKU information
- No user interaction required

**When to use**: For automated/production runs, scheduled forecasting, or when you want
to update the database with fresh forecasts.

**Run with**: `python src/machine_learning/script_final.py`

### 2. `script_main.py` - Interactive Model Training & Predictions

**Purpose**: Interactive script for training models and visualizing individual MLB
predictions.

**What it does**:

- Loads data (cached or fresh from database)
- Processes data and creates features
- Trains XGBoost models for each MLB (or loads existing models)
- Generates sample time series plots
- Allows interactive MLB selection for prediction visualization (displays SKU for
  reference)
- Saves prediction plots as PNG files

**When to use**: For model development, training new models, or analyzing individual MLB
predictions.

**Run with**: `python src/machine_learning/script_main.py`

### 3. `script_main_forecast.py` - Interactive Forecasting

**Purpose**: Interactive script for generating and visualizing forecasts with user
input.

**What it does**:

- Loads data (cached or fresh from database)
- Processes data and creates features
- Generates 30-day forecasts for all MLBs (or loads existing forecasts)
- Allows interactive MLB selection for forecast visualization (displays SKU for
  reference)
- Saves forecast plots as HTML files
- Saves forecasts to database

**When to use**: For interactive forecasting sessions, exploring forecasts for specific
MLBs, or generating forecast visualizations.

**Run with**: `python src/machine_learning/script_main_forecast.py`

### 4. `script_webapp.py` - Web Dashboard

**Purpose**: Web-based dashboard for viewing forecasts.

**What it does**:

- Launches a web interface at http://127.0.0.1:8050
- Displays interactive forecast charts and data tables
- Includes data caching for performance
- Allows filtering by MLB and SKU with date range

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

**Script Behavior**:

- `script_final.py`: Always imports fresh data from database (no caching)
- `script_main.py` and `script_main_forecast.py`: Edit the boolean flags in each script:

```python
# In script_main.py and script_main_forecast.py
import_data = True  # Set to True to fetch fresh data (recommended)
import_regressor = True  # Set to False to train new models
```

## Data Flow

1. **Data Import**: Raw sales data from PostgreSQL (`public.view_enrico`) - requires
   `mlb` column
1. **Data Processing**: Cleaning and time series feature creation, grouped by MLB
1. **Model Training**: XGBoost regressors trained per MLB (with SKU preserved for
   reference)
1. **Forecasting**: Multi-step direct forecasting using MLB as unique identifier
1. **Storage**: Results saved to database and local files with both MLB and SKU
   information
1. **Visualization**: Web dashboard and plot generation

## Key File Locations

- **Configuration**: `src/machine_learning/config.py` - Main configuration
- **Models**: `src/machine_learning/bld/mlb_regressors.pkl`
- **Forecasts**: `src/machine_learning/bld/mlb_forecast.pkl`
- **Data**: `src/machine_learning/data/raw_sql.csv`
- **Plots**: `src/machine_learning/bld/*.html` and `*.png`

## Database Configuration

- **Host**: 172.27.40.210:5432
- **Database**: "Mercado Livre"
- **Input View**: `public.view_enrico`
- **Output Table**: `public.mlb_forecasts_90_days` (contains both MLB and SKU columns)

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

## MLB vs SKU Architecture

**Important**: This system has been redesigned to use MLB (Mercado Livre identifiers) as
the primary forecasting unit:

- **MLB as Primary ID**: All forecasting, model training, and processing uses MLB as the
  unique identifier
- **SKU Preservation**: SKU information is maintained throughout the pipeline for
  reference and display
- **Relationship**: Many SKUs can share the same MLB code, but each MLB maps to exactly
  one SKU
- **Database Requirements**: The input view (`public.view_enrico`) **must include an
  `mlb` column**
- **Output Structure**: Forecasts are saved with both MLB and SKU columns for complete
  traceability

## Development Notes

- Models are trained per MLB individually using XGBoost (SKU preserved for reference)
- The system supports both model training and loading pre-trained models
- All scripts include comprehensive error handling and logging
- The web application includes data caching for improved performance
- Forecasts are generated using Multi Step Direct Forecasting methodology
- **Data Import**: Uses direct database import (no chunking) to ensure all recent data
  is captured
- **Timezone Handling**: Properly converts timezone-aware dates to avoid data processing
  issues
- **Prediction Rounding**: Uses normal rounding (5.4→5, 5.6→6) instead of ceiling for
  more accurate forecasts
- **Data Freshness**: Validates data recency to prevent forecasting with stale MLB data
- **MLB-Based Processing**: All data processing, feature creation, and aggregation is
  done by MLB
- **Interactive Scripts**: When prompted for input, enter MLB codes (SKU will be
  displayed for reference)
