"""
Interactive forecasting script with user input for SKU selection.
Allows users to choose between importing fresh data or using cached data/forecasts.
"""

import logging
import pandas as pd
import plotly.express as px

from config import AppConfig
from database_utils import db_manager
from estimation.config_ml import BLD
from data_management.clean_sql_data import process_sales_data
from data_management.feature_creation import create_time_series_features
from estimation.model_forecast import save_regressors, forecast_future_sales_direct
from estimation.plot import print_available_skus

# Configure plotting backend
pd.options.plotting.backend = "matplotlib"

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def validate_data_freshness(data: pd.DataFrame, max_age_days: int = 7) -> None:
    """
    Validate that the cached data is fresh enough to use.
    Issues warnings if data is stale.

    Args:
        data: DataFrame containing the raw data
        max_age_days: Maximum acceptable age of data in days
    """
    try:
        # Determine date column
        date_col = None
        for col in ["date_created", "date"]:
            if col in data.columns:
                date_col = col
                break

        if not date_col:
            logger.warning(
                "No date column found in cached data - cannot validate freshness"
            )
            return

        # Parse dates and find the most recent
        dates = pd.to_datetime(data[date_col], errors="coerce")
        most_recent = dates.max()

        if pd.isna(most_recent):
            logger.warning("Could not parse dates in cached data")
            return

        # Calculate age of data
        today = pd.Timestamp.now().normalize()
        days_old = (today - most_recent.normalize()).days

        logger.info(
            f"Cached data age: {days_old} days (most recent: {most_recent.date()})"
        )

        if days_old > max_age_days:
            logger.warning(
                f"⚠️  STALE DATA WARNING: Cached data is {days_old} days old! "
                f"This may cause all MLBs to be marked as inactive. "
                f"Consider running with import_data=True or running refresh_data.py"
            )
        elif days_old > 2:
            logger.info(
                f"📅 Cached data is {days_old} days old - still usable but consider refreshing"
            )
        else:
            logger.info("✅ Cached data is fresh")

    except Exception as e:
        logger.warning(f"Error validating data freshness: {e}")


def load_or_import_data(import_data: bool) -> pd.DataFrame:
    """Load data from file or import fresh data from database."""
    if import_data:
        logger.info("Importing fresh data from SQL...")
        data = db_manager.import_data_from_sql()

        # Save to CSV for future use
        data.to_csv(AppConfig.RAW_DATA_FILE, index=False)
        logger.info("Data imported and cached successfully!")
        return data
    else:
        logger.info("Loading cached data from CSV...")

        # Check if cached file exists
        if not AppConfig.RAW_DATA_FILE.exists():
            logger.warning(f"Cached data file not found: {AppConfig.RAW_DATA_FILE}")
            logger.info("Falling back to importing fresh data from database...")
            data = db_manager.import_data_from_sql()
            data.to_csv(AppConfig.RAW_DATA_FILE, index=False)
            return data

        data = pd.read_csv(AppConfig.RAW_DATA_FILE, engine="pyarrow")

        # Validate data freshness
        validate_data_freshness(data)

        return data


def load_or_generate_forecasts(
    import_forecast: bool, feature_data: pd.DataFrame
) -> dict:
    """Load existing forecasts or generate new ones."""
    if import_forecast and AppConfig.MLB_FORECAST_FILE.exists():
        logger.info(f"Loading forecasts from {AppConfig.MLB_FORECAST_FILE}...")
        return pd.read_pickle(AppConfig.MLB_FORECAST_FILE)
    else:
        logger.info(f"Generating new {AppConfig.FORECAST_DAYS}-day forecasts...")
        mlb_forecast = forecast_future_sales_direct(
            feature_data, AppConfig.FORECAST_DAYS
        )

        # Save forecasts for future use
        save_regressors(mlb_forecast, AppConfig.MLB_FORECAST_FILE)
        logger.info("Forecasting complete and saved!")
        return mlb_forecast


def interactive_mlb_selection(mlb_forecast: dict) -> None:
    """Handle interactive MLB selection and visualization."""
    print_available_skus(
        AppConfig.MLB_FORECAST_FILE
    )  # Still uses print_available_skus function

    # Get user input for MLB
    mlb = input("Please enter the MLB for predictions: ").strip()

    # Generate and save visualization
    if mlb in mlb_forecast:
        forecast_df, sku = mlb_forecast[mlb]

        if forecast_df is not None and not forecast_df.empty:
            fig = px.line(forecast_df, title=f"Forecast for MLB {mlb} (SKU {sku})")
            html_file = BLD / f"{mlb}_forecast.html"
            fig.write_html(html_file)
            logger.info(f"Forecast plotted and saved as HTML: {html_file}")
        else:
            logger.warning(f"No forecast available for MLB {mlb}")
    else:
        logger.warning(f"No forecast available for MLB {mlb}")


def main():
    """Main execution function for interactive forecasting."""
    try:
        # Configuration flags
        import_data = True
        import_forecast = True

        # Load or import data
        data = load_or_import_data(import_data)

        # Process data
        logger.info("Processing sales data...")
        clean_data = process_sales_data(data)

        logger.info("Creating time series features...")
        feature_data = create_time_series_features(clean_data)

        # Save feature data for debugging/analysis
        feature_data.to_csv(AppConfig.FEATURE_DATA_FILE, index=False)
        logger.info("Data processing complete!")

        # Load or generate forecasts
        mlb_forecast = load_or_generate_forecasts(import_forecast, feature_data)

        # Interactive MLB selection and visualization
        interactive_mlb_selection(mlb_forecast)

        # Save forecasts to database
        logger.info("Saving forecasts to remote SQL database...")
        db_manager.save_forecasts_to_sql(mlb_forecast)
        logger.info("Forecasts saved to database successfully!")

    except Exception as e:
        logger.error(f"Script failed with error: {e}")
        raise
    finally:
        db_manager.close_connection()


if __name__ == "__main__":
    main()
