"""
Final production script for sales forecasting.
This script imports data, processes it, generates forecasts, and saves to database.
"""

import logging
import pandas as pd

from config import AppConfig
from database_utils import db_manager, validate_data_freshness
from data_management.clean_sql_data import process_sales_data
from data_management.feature_creation import create_time_series_features
from estimation.model_forecast import forecast_future_sales_direct

# Configure plotting backend
pd.options.plotting.backend = "matplotlib"

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main execution function for the forecasting pipeline."""
    try:
        # Test database connection
        if not db_manager.test_connection():
            logger.error("Database connection failed. Check your configuration.")
            return

        # Import data from SQL
        logger.info("Starting data import from SQL...")
        data = db_manager.import_data_from_sql()
        logger.info("Data imported successfully!")

        # Process and clean data
        logger.info("Processing sales data...")
        clean_data = process_sales_data(data)

        # Create time series features
        logger.info("Creating time series features...")
        feature_data = create_time_series_features(clean_data)
        logger.info("Data processing complete!")

        # Quick data freshness check
        validate_data_freshness(feature_data)

        # Generate forecasts
        logger.info(f"Generating {AppConfig.FORECAST_DAYS_LONG}-day forecasts...")
        sku_forecast = forecast_future_sales_direct(
            feature_data, AppConfig.FORECAST_DAYS_LONG
        )
        logger.info("Forecasting complete!")

        # Save forecasts to database
        logger.info("Saving forecasts to remote SQL database...")
        db_manager.save_forecasts_to_sql(sku_forecast)
        logger.info("Forecasts saved successfully!")

    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        raise
    finally:
        # Clean up database connection
        db_manager.close_connection()


if __name__ == "__main__":
    main()
