"""
Incremental forecasting pipeline for continuous learning.
This script uses incremental model updates with XGBoost continuation training
to update models with new data without full retraining.
"""

import logging
import numpy as np
import pandas as pd
import sys
import time
from pathlib import Path

# Add src path to import the modules
sys.path.append(str(Path(__file__).parent.parent))

from config import AppConfig
from database_utils import db_manager, validate_data_freshness
from data_management.clean_sql_data import process_sales_data
from data_management.feature_creation import create_time_series_features
from data_management.metadata_tracker import create_metadata_table, log_pipeline_run
from data_management.data_merger import (
    merge_with_historical,
    save_merged_data,
    get_date_range_info,
)
from estimation.model_forecast import (
    update_mlb_models_incremental,
    forecast_future_sales_direct,
)
from estimation.model_storage import save_models, load_models, archive_models

# Configure plotting backend
pd.options.plotting.backend = "matplotlib"

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def generate_forecasts_for_existing_models(
    updated_models: dict, feature_data: pd.DataFrame, forecast_days: int
) -> tuple[dict, dict]:
    """
    Generate forecasts only for MLBs that have existing/updated models.
    This avoids iterating through the entire dataset when we only want to forecast
    for specific MLBs we have models for.

    Args:
        updated_models: Dictionary of MLB -> trained model
        feature_data: Full dataset with all MLBs
        forecast_days: Number of days to forecast

    Returns:
        tuple[dict, dict]: (mlb_forecasts, mlb_models) same format as other forecast functions
    """
    logger = logging.getLogger(__name__)
    mlb_forecasts = {}
    mlb_models = {}

    logger.info(
        f"Generating targeted forecasts for {len(updated_models)} specific MLBs"
    )

    for mlb, model in updated_models.items():
        try:
            # Get data for this specific MLB
            mlb_data = feature_data[feature_data["mlb"] == mlb]
            if len(mlb_data) == 0:
                logger.warning(f"No data found for MLB {mlb}, skipping forecast")
                continue

            # Get the SKU for this MLB
            sku = mlb_data["sku"].iloc[0] if "sku" in mlb_data.columns else None

            # Generate forecast dates
            today = pd.Timestamp.now().normalize()
            future_start = today + pd.Timedelta(days=1)
            future_dates = pd.date_range(
                start=future_start, periods=forecast_days, freq="D"
            )

            # For MultiOutputRegressor models, use them directly
            if hasattr(model, "estimators_"):
                # Use the last row of features for prediction
                FEATURES = ["day_of_week", "day_of_month", "rolling_mean_3", "lag_1"]
                if not all(feature in mlb_data.columns for feature in FEATURES):
                    logger.warning(f"Missing required features for MLB {mlb}, skipping")
                    continue

                last_features = mlb_data.iloc[-1][FEATURES].values.reshape(1, -1)
                predictions = model.predict(last_features)[
                    0
                ]  # MultiOutput returns array of arrays
            else:
                # For regular models, also use direct prediction
                FEATURES = ["day_of_week", "day_of_month", "rolling_mean_3", "lag_1"]
                if not all(feature in mlb_data.columns for feature in FEATURES):
                    logger.warning(f"Missing required features for MLB {mlb}, skipping")
                    continue

                last_features = mlb_data.iloc[-1][FEATURES].values.reshape(1, -1)
                predictions = model.predict(last_features)
                if len(predictions.shape) == 1:
                    predictions = predictions.reshape(1, -1)[0]

            # Ensure we have the right number of predictions
            if len(predictions) != forecast_days:
                logger.warning(
                    f"Model for MLB {mlb} returned {len(predictions)} predictions, expected {forecast_days}"
                )
                # Pad or truncate as needed
                if len(predictions) < forecast_days:
                    # Repeat last prediction
                    last_pred = predictions[-1] if len(predictions) > 0 else 0
                    predictions = list(predictions) + [last_pred] * (
                        forecast_days - len(predictions)
                    )
                else:
                    predictions = predictions[:forecast_days]

            # Round predictions to nearest integers (products sold in whole units)
            predictions = np.round(predictions).astype(int)

            # Create forecast DataFrame
            forecast_df = pd.DataFrame({"prediction": predictions}, index=future_dates)

            # Store forecast and model
            mlb_forecasts[mlb] = (forecast_df, sku)
            mlb_models[mlb] = model

            logger.debug(f"Generated forecast for MLB {mlb} (SKU {sku})")

        except Exception as e:
            logger.error(f"Failed to generate forecast for MLB {mlb}: {e}")
            continue

    logger.info(f"Successfully generated forecasts for {len(mlb_forecasts)} MLBs")
    return mlb_forecasts, mlb_models


def main():
    """Main execution function for the incremental forecasting pipeline."""
    start_time = time.time()
    run_type = "incremental"
    records_processed = 0
    models_updated = 0
    error_message = None
    use_incremental = False

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

        # Check if existing models are available for incremental training
        existing_models = {}
        if AppConfig.MLB_REGRESSORS_FILE.exists():
            try:
                logger.info(
                    f"Loading existing models from {AppConfig.MLB_REGRESSORS_FILE}"
                )
                existing_models = load_models(AppConfig.MLB_REGRESSORS_FILE)
                logger.info(f"Loaded {len(existing_models)} existing models")
                use_incremental = True
            except Exception as e:
                logger.warning(f"Failed to load existing models: {e}")
                logger.info("Proceeding with full training instead")
                use_incremental = False
        else:
            logger.info("No existing models found, will perform full training")
            use_incremental = False

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

        # Get date range info for logging
        date_info = get_date_range_info(feature_data)
        if date_info["min_date"] is not None and date_info["max_date"] is not None:
            logger.info(
                f"Data spans from {date_info['min_date']} to {date_info['max_date']} "
                f"({date_info['date_span_days']} days, {date_info['record_count']:,} records)"
            )
        else:
            logger.info(
                f"Processing {date_info['record_count']:,} records (date range not available)"
            )

        # Decide between incremental update or full training
        if use_incremental and existing_models:
            logger.info("=== INCREMENTAL TRAINING MODE ===")
            logger.info(
                f"Updating {len(existing_models)} existing models with new data"
            )

            try:
                # Backup existing models before updating
                archive_dir = AppConfig.MLB_REGRESSORS_FILE.parent / "archive"
                archive_models(AppConfig.MLB_REGRESSORS_FILE, archive_dir)

                # Perform incremental model updates
                logger.info("Starting incremental model updates...")
                updated_models = update_mlb_models_incremental(
                    existing_models,
                    feature_data,
                    additional_rounds=50,  # Conservative number for incremental updates
                )
                models_updated = len(updated_models)
                logger.info(f"Incremental update completed for {models_updated} models")

                # Generate forecasts using targeted approach for updated models only
                logger.info(
                    f"Generating {AppConfig.FORECAST_DAYS_LONG}-day forecasts for {len(updated_models)} updated models..."
                )
                mlb_forecast, mlb_models = generate_forecasts_for_existing_models(
                    updated_models, feature_data, AppConfig.FORECAST_DAYS_LONG
                )
                logger.info("Targeted forecasting with updated models complete!")

                logger.info("Successfully completed incremental training pipeline")

            except Exception as incremental_error:
                logger.error(f"Incremental training failed: {incremental_error}")
                logger.info("Falling back to full training...")

                # Fallback to full training
                mlb_forecast, mlb_models = forecast_future_sales_direct(
                    feature_data, AppConfig.FORECAST_DAYS_LONG
                )
                models_updated = len(mlb_models)
                run_type = "full_fallback"
                logger.info("Fallback full training completed successfully")

        else:
            logger.info("=== FULL TRAINING MODE ===")
            logger.info("Training all models from scratch")

            # Generate forecasts and train models from scratch
            logger.info(f"Generating {AppConfig.FORECAST_DAYS_LONG}-day forecasts...")
            mlb_forecast, mlb_models = forecast_future_sales_direct(
                feature_data, AppConfig.FORECAST_DAYS_LONG
            )
            models_updated = len(mlb_models)
            run_type = "full"
            logger.info("Full training completed successfully")

        # Save forecasts to database
        logger.info("Saving forecasts to remote SQL database...")
        db_manager.save_forecasts_to_sql(mlb_forecast)
        logger.info("Forecasts saved successfully!")

        # Save trained/updated models for continuous learning
        logger.info("Saving trained models for continuous learning...")
        save_models(mlb_models, AppConfig.MLB_REGRESSORS_FILE)
        logger.info(
            f"Saved {len(mlb_models)} trained models to {AppConfig.MLB_REGRESSORS_FILE}"
        )

        # Calculate execution time and log successful completion
        end_time = time.time()
        execution_time = end_time - start_time

        logger.info("INCREMENTAL PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info(f"Training mode: {run_type}")
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
        logger.error(f"Incremental pipeline failed with error: {error_message}")

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


def main_with_date_filter(since_date: str):
    """
    Alternative main function that uses date-filtered data import for true incremental updates.

    Args:
        since_date: Date string in YYYY-MM-DD format to import data since this date
    """
    start_time = time.time()
    run_type = "incremental_filtered"
    records_processed = 0
    models_updated = 0
    error_message = None

    try:
        logger.info(
            f"Starting incremental pipeline with date filter: since {since_date}"
        )

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

        # Load existing models (required for incremental updates)
        if not AppConfig.MLB_REGRESSORS_FILE.exists():
            error_message = f"No existing models found at {AppConfig.MLB_REGRESSORS_FILE}. Run full training first."
            logger.error(error_message)
            raise FileNotFoundError(error_message)

        logger.info(f"Loading existing models from {AppConfig.MLB_REGRESSORS_FILE}")
        existing_models = load_models(AppConfig.MLB_REGRESSORS_FILE)
        logger.info(f"Loaded {len(existing_models)} existing models")

        # Import new data since specified date using date-filtered import
        from data_management.import_SQL import import_data_from_sql_since_date
        from config import DatabaseConfig

        logger.info(f"Importing new data since {since_date}...")
        new_data = import_data_from_sql_since_date(
            user=DatabaseConfig.USER,
            password=DatabaseConfig.PASSWORD,
            host=DatabaseConfig.HOST,
            port=DatabaseConfig.PORT,
            dbname=DatabaseConfig.DBNAME,
            view=DatabaseConfig.VIEW,
            since_date=since_date,
        )

        if new_data.empty:
            logger.info(
                "No new data found since the specified date. Pipeline completed with no updates."
            )

            # Log successful but no-op run
            log_pipeline_run(
                run_type=run_type,
                status="success",
                records_processed=0,
                models_updated=0,
                run_duration_seconds=time.time() - start_time,
            )
            return

        records_processed = len(new_data)
        logger.info(f"Imported {records_processed:,} new records since {since_date}")

        # Merge with historical data if needed for feature creation
        historical_data_path = AppConfig.RAW_DATA_FILE
        if historical_data_path.exists():
            logger.info(
                "Merging new data with historical data for feature consistency..."
            )
            merged_data = merge_with_historical(new_data, historical_data_path)
            logger.info(f"Merged dataset contains {len(merged_data):,} total records")
        else:
            logger.warning("No historical data file found, using only new data")
            merged_data = new_data

        # Process and clean data
        logger.info("Processing sales data...")
        clean_data = process_sales_data(merged_data)

        # Create time series features
        logger.info("Creating time series features...")
        feature_data = create_time_series_features(clean_data)
        logger.info("Data processing complete!")

        # Perform incremental model updates with new data only
        logger.info("Starting incremental model updates with new data...")

        # Backup existing models before updating
        archive_dir = AppConfig.MLB_REGRESSORS_FILE.parent / "archive"
        archive_models(AppConfig.MLB_REGRESSORS_FILE, archive_dir)

        # Filter feature_data to only include MLBs that have new data
        mlbs_with_new_data = new_data["mlb"].unique()
        incremental_data = feature_data[feature_data["mlb"].isin(mlbs_with_new_data)]
        logger.info(
            f"Performing incremental updates for {len(mlbs_with_new_data)} MLBs with new data"
        )

        updated_models = update_mlb_models_incremental(
            existing_models, incremental_data, additional_rounds=50
        )
        models_updated = len(updated_models)

        # Generate forecasts for updated models
        logger.info(
            f"Generating {AppConfig.FORECAST_DAYS_LONG}-day forecasts for updated models..."
        )
        mlb_forecast = {}

        for mlb, model in updated_models.items():
            mlb_data = feature_data[feature_data["mlb"] == mlb]
            if len(mlb_data) > 0:
                # Use the model to generate forecast for this MLB
                FEATURES = ["day_of_week", "day_of_month", "rolling_mean_3", "lag_1"]
                last_features = mlb_data.iloc[-1][FEATURES].values.reshape(1, -1)  # type: ignore
                predictions = model.predict(last_features)[0]  # type: ignore

                # Build forecast DataFrame
                today = pd.Timestamp.now().normalize()
                future_start = today + pd.Timedelta(days=1)
                future_dates = pd.date_range(
                    start=future_start, periods=AppConfig.FORECAST_DAYS_LONG, freq="D"
                )

                # Round predictions to nearest integers (products sold in whole units)
                predictions = np.round(predictions).astype(int)

                forecast_df = pd.DataFrame(
                    {"prediction": predictions}, index=future_dates
                )
                sku = mlb_data["sku"].iloc[0] if "sku" in mlb_data.columns else None
                mlb_forecast[mlb] = (forecast_df, sku)

        logger.info(f"Generated forecasts for {len(mlb_forecast)} MLBs")

        # Save forecasts to database
        if mlb_forecast:
            logger.info("Saving forecasts to remote SQL database...")
            db_manager.save_forecasts_to_sql(mlb_forecast)
            logger.info("Forecasts saved successfully!")

        # Save updated models
        logger.info("Saving updated models for continuous learning...")
        save_models(updated_models, AppConfig.MLB_REGRESSORS_FILE)
        logger.info(
            f"Saved {len(updated_models)} updated models to {AppConfig.MLB_REGRESSORS_FILE}"
        )

        # Update historical data file with new data
        save_merged_data(merged_data, historical_data_path)
        logger.info(
            f"Updated historical data file with {records_processed:,} new records"
        )

        # Calculate execution time and log successful completion
        end_time = time.time()
        execution_time = end_time - start_time

        logger.info("INCREMENTAL PIPELINE WITH DATE FILTER COMPLETED SUCCESSFULLY!")
        logger.info(f"Execution time: {execution_time:.1f} seconds")
        logger.info(f"New records processed: {records_processed:,}")
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
        logger.error(f"Incremental pipeline with date filter failed: {error_message}")

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
    import sys

    # Check if date filter argument is provided
    if len(sys.argv) > 1:
        since_date = sys.argv[1]
        logger.info(f"Running incremental pipeline with date filter: {since_date}")
        main_with_date_filter(since_date)
    else:
        logger.info("Running standard incremental pipeline")
        main()
