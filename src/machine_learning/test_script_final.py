"""
Test version of the final production script for sales forecasting.
This script imports data, processes it, generates forecasts for a limited number of MLBs,
and saves to test database table - designed for fast testing during development.
"""

import logging
import pandas as pd
import time

from config import AppConfig, DatabaseConfig
from database_utils import db_manager, validate_data_freshness
from data_management.clean_sql_data import process_sales_data
from data_management.feature_creation import create_time_series_features
from estimation.model_forecast import forecast_future_sales_direct_limited
from estimation.model_storage import save_models

# Configure plotting backend
pd.options.plotting.backend = "matplotlib"

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main execution function for the test forecasting pipeline."""
    start_time = time.time()

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

        # Generate forecasts and train models (LIMITED VERSION)
        logger.info(
            f"Generating {AppConfig.FORECAST_DAYS_LONG}-day forecasts for up to {AppConfig.TEST_MLB_COUNT} MLBs..."
        )
        mlb_forecast, mlb_models = forecast_future_sales_direct_limited(
            feature_data, AppConfig.FORECAST_DAYS_LONG, AppConfig.TEST_MLB_COUNT
        )
        logger.info("Forecasting complete!")

        # Save forecasts to TEST database table
        logger.info(
            f"Saving forecasts to test SQL database ({DatabaseConfig.TEST_FORECAST_TABLE})..."
        )
        db_manager.save_forecasts_to_sql(
            mlb_forecast, DatabaseConfig.TEST_FORECAST_TABLE
        )
        logger.info("Forecasts saved successfully!")

        # Save trained models for continuous learning (test files)
        test_models_file = (
            AppConfig.MLB_REGRESSORS_FILE.parent / "test_mlb_regressors.pkl"
        )
        logger.info("Saving trained models for continuous learning...")
        save_models(mlb_models, test_models_file)
        logger.info(f"Saved {len(mlb_models)} trained models to {test_models_file}")

        # Save forecast data to pickle for testing
        test_forecast_file = (
            AppConfig.MLB_FORECAST_FILE.parent / "test_mlb_forecast.pkl"
        )
        import pickle

        with test_forecast_file.open("wb") as f:
            pickle.dump(mlb_forecast, f)
        logger.info(f"Saved forecast data to {test_forecast_file}")

        # Calculate and log execution time
        end_time = time.time()
        execution_time = end_time - start_time
        minutes = int(execution_time // 60)
        seconds = int(execution_time % 60)

        logger.info("TEST PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info(
            f"Execution time: {minutes}m {seconds}s ({execution_time:.1f}s total)"
        )
        logger.info(
            f"Processed {len(mlb_models)} MLBs out of {AppConfig.TEST_MLB_COUNT} requested"
        )

    except Exception as e:
        logger.error(f"Test pipeline failed with error: {e}")
        raise
    finally:
        # Clean up database connection
        db_manager.close_connection()


if __name__ == "__main__":
    main()
