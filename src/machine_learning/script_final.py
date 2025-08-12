"""
Final production script for sales forecasting.
This script imports data, processes it, generates forecasts, and saves to database.
"""

import logging
import pandas as pd
import time

from config import AppConfig
from database_utils import db_manager, validate_data_freshness
from data_management.clean_sql_data import process_sales_data
from data_management.feature_creation import create_time_series_features
from data_management.metadata_tracker import create_metadata_table, log_pipeline_run
from estimation.model_forecast import forecast_future_sales_direct
from estimation.model_storage import save_models

# Configure plotting backend
pd.options.plotting.backend = "matplotlib"

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main execution function for the forecasting pipeline."""
    start_time = time.time()
    run_type = "full"
    records_processed = 0
    models_updated = 0
    error_message = None

    try:
        # Create metadata table if it doesn't exist
        logger.info("Initializing metadata tracking...")
        if not create_metadata_table():
            logger.warning(
                "Failed to create metadata table, continuing without metadata tracking"
            )

        # Test database connection
        if not db_manager.test_connection():
            error_message = "Database connection failed"
            logger.error(error_message)
            return

        # Import data from SQL
        logger.info("Starting data import from SQL...")
        data = db_manager.import_data_from_sql()
        records_processed = len(data)
        logger.info(f"Data imported successfully! ({records_processed:,} records)")

        # Process and clean data
        logger.info("Processing sales data...")
        clean_data = process_sales_data(data)

        # Create time series features
        logger.info("Creating time series features...")
        feature_data = create_time_series_features(clean_data)
        logger.info("Data processing complete!")

        # Quick data freshness check
        validate_data_freshness(feature_data)

        # Generate forecasts and train models
        logger.info(f"Generating {AppConfig.FORECAST_DAYS_LONG}-day forecasts...")
        mlb_forecast, mlb_models = forecast_future_sales_direct(
            feature_data, AppConfig.FORECAST_DAYS_LONG
        )
        models_updated = len(mlb_models)
        logger.info("Forecasting complete!")

        # Save forecasts to database
        logger.info("Saving forecasts to remote SQL database...")
        db_manager.save_forecasts_to_sql(mlb_forecast)
        logger.info("Forecasts saved successfully!")

        # Save trained models for continuous learning
        logger.info("Saving trained models for continuous learning...")
        save_models(mlb_models, AppConfig.MLB_REGRESSORS_FILE)
        logger.info(
            f"Saved {len(mlb_models)} trained models to {AppConfig.MLB_REGRESSORS_FILE}"
        )

        # Calculate execution time and log successful completion
        end_time = time.time()
        execution_time = end_time - start_time

        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info(f"Execution time: {execution_time:.1f} seconds")
        logger.info(f"Records processed: {records_processed:,}")
        logger.info(f"Models updated: {models_updated}")

        # Log successful pipeline run
        log_pipeline_run(
            run_type=run_type,
            status="success",
            records_processed=records_processed,
            models_updated=models_updated,
            run_duration_seconds=execution_time,
        )

    except Exception as e:
        error_message = str(e)
        logger.error(f"Pipeline failed with error: {error_message}")

        # Log failed pipeline run
        end_time = time.time()
        execution_time = end_time - start_time
        log_pipeline_run(
            run_type=run_type,
            status="failed",
            records_processed=records_processed,
            models_updated=models_updated,
            error_message=error_message,
            run_duration_seconds=execution_time,
        )
        raise
    finally:
        # Clean up database connection
        db_manager.close_connection()


if __name__ == "__main__":
    main()
