"""
Test version of the incremental forecasting pipeline for continuous learning.
This script uses incremental model updates with XGBoost continuation training
to update models with new data without full retraining.
Limited to 5 MLBs for testing purposes to avoid large file size issues.
"""

import logging
import numpy as np
import pandas as pd
import sys
import time
import xgboost as xgb
from pathlib import Path

# Add src path to import the modules
sys.path.append(str(Path(__file__).parent.parent))

from config import AppConfig, DatabaseConfig
from database_utils import db_manager, validate_data_freshness
from data_management.clean_sql_data import process_sales_data
from data_management.feature_creation import create_time_series_features
from data_management.metadata_tracker import create_metadata_table, log_pipeline_run
from data_management.data_merger import (
    get_date_range_info,
)
from estimation.model_forecast import (
    forecast_future_sales_direct_limited,
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


def update_mlb_models_incremental_limited(
    existing_models: dict,
    new_data: pd.DataFrame,
    max_mlbs: int = 5,
    additional_rounds: int = 50,
) -> dict:
    """
    Limited version of incremental model update for testing.
    Updates up to max_mlbs models with new data using XGBoost continuation training.

    Args:
        existing_models: Dictionary of existing trained models per MLB
        new_data: DataFrame with new data for training
        max_mlbs: Maximum number of MLBs to update (for testing)
        additional_rounds: Number of additional boosting rounds for incremental training

    Returns:
        Dictionary of updated models
    """
    logger = logging.getLogger(__name__)
    updated_models = {}
    processed_count = 0

    FEATURES = ["day_of_week", "day_of_month", "rolling_mean_3", "lag_1"]
    TARGET = "quant"

    # Get MLBs that have both existing models and new data
    available_mlbs = []
    for mlb in existing_models.keys():
        mlb_data = new_data[new_data["mlb"] == mlb]
        if len(mlb_data) >= 10:  # Need minimum data for incremental training
            available_mlbs.append(mlb)
        if len(available_mlbs) >= max_mlbs:
            break

    logger.info(
        f"Limiting incremental update to {len(available_mlbs)} MLBs with sufficient data: {available_mlbs[:max_mlbs]}"
    )

    for mlb in available_mlbs[:max_mlbs]:
        if processed_count >= max_mlbs:
            logger.info(
                f"Reached maximum MLB limit ({max_mlbs}), stopping incremental updates"
            )
            break

        # Get data for this MLB
        mlb_data = new_data[new_data["mlb"] == mlb].copy()

        try:
            # Prepare features and target
            if not all(feature in mlb_data.columns for feature in FEATURES):
                logger.warning(f"Missing required features for MLB {mlb}, skipping")
                continue

            X_new = mlb_data[FEATURES].values
            y_new = mlb_data[TARGET].values

            # Get existing model
            existing_model = existing_models[mlb]

            # Create new model with same parameters for continuation training
            if hasattr(existing_model, "estimators_"):
                # It's a MultiOutputRegressor, get the base estimator parameters
                base_estimator = existing_model.estimators_[0]
                updated_model = xgb.XGBRegressor(
                    n_estimators=base_estimator.n_estimators + additional_rounds,
                    max_depth=base_estimator.max_depth,
                    learning_rate=base_estimator.learning_rate,
                    random_state=getattr(base_estimator, "random_state", 42),
                )
            else:
                # It's a regular XGBRegressor
                updated_model = xgb.XGBRegressor(
                    n_estimators=existing_model.n_estimators + additional_rounds,
                    max_depth=existing_model.max_depth,
                    learning_rate=existing_model.learning_rate,
                    random_state=getattr(existing_model, "random_state", 42),
                )

            # Continue training from existing model
            if hasattr(existing_model, "estimators_"):
                # For MultiOutputRegressor, we need to handle it differently
                logger.warning(
                    f"MLB {mlb} has MultiOutputRegressor - using existing model without incremental update"
                )
                updated_models[mlb] = existing_model
            else:
                # For regular XGBRegressor, use continuation training
                updated_model.fit(
                    X_new,
                    y_new,
                    xgb_model=existing_model,  # This enables continuation training
                )
                updated_models[mlb] = updated_model

            processed_count += 1
            logger.info(
                f"Successfully processed model for MLB {mlb} ({processed_count}/{max_mlbs})"
            )

        except Exception as e:
            logger.error(f"Failed to update model for MLB {mlb}: {e}")
            # Keep the original model
            updated_models[mlb] = existing_models[mlb]
            processed_count += 1

    logger.info(f"Completed incremental updates for {len(updated_models)} MLBs")
    return updated_models


def main():
    """Main execution function for the test incremental forecasting pipeline."""
    start_time = time.time()
    run_type = "test_incremental"
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

        # Check if existing test models are available for incremental training
        existing_models = {}
        test_models_file = (
            AppConfig.MLB_REGRESSORS_FILE.parent / "test_mlb_regressors.pkl"
        )

        if test_models_file.exists():
            try:
                logger.info(f"Loading existing test models from {test_models_file}")
                existing_models = load_models(test_models_file)
                logger.info(f"Loaded {len(existing_models)} existing test models")
                use_incremental = True
            except Exception as e:
                logger.warning(f"Failed to load existing test models: {e}")
                logger.info("Proceeding with full training instead")
                use_incremental = False
        else:
            logger.info(
                "No existing test models found, will perform limited full training"
            )
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
            logger.info("=== TEST INCREMENTAL TRAINING MODE ===")
            logger.info(
                f"Updating up to {AppConfig.TEST_MLB_COUNT} existing models with new data"
            )

            try:
                # Backup existing test models before updating
                archive_dir = test_models_file.parent / "archive"
                archive_models(test_models_file, archive_dir)

                # Perform limited incremental model updates
                logger.info("Starting limited incremental model updates...")
                updated_models = update_mlb_models_incremental_limited(
                    existing_models,
                    feature_data,
                    max_mlbs=AppConfig.TEST_MLB_COUNT,
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
                logger.info("Successfully completed test incremental training pipeline")

            except Exception as incremental_error:
                logger.error(f"Test incremental training failed: {incremental_error}")
                logger.info("Falling back to limited full training...")

                # Fallback to limited full training
                mlb_forecast, mlb_models = forecast_future_sales_direct_limited(
                    feature_data, AppConfig.FORECAST_DAYS_LONG, AppConfig.TEST_MLB_COUNT
                )
                models_updated = len(mlb_models)
                run_type = "test_full_fallback"
                logger.info("Fallback limited full training completed successfully")

        else:
            logger.info("=== TEST FULL TRAINING MODE ===")
            logger.info(
                f"Training up to {AppConfig.TEST_MLB_COUNT} models from scratch"
            )

            # Generate forecasts and train models from scratch (LIMITED)
            logger.info(
                f"Generating {AppConfig.FORECAST_DAYS_LONG}-day forecasts (limited to {AppConfig.TEST_MLB_COUNT} MLBs)..."
            )
            mlb_forecast, mlb_models = forecast_future_sales_direct_limited(
                feature_data, AppConfig.FORECAST_DAYS_LONG, AppConfig.TEST_MLB_COUNT
            )
            models_updated = len(mlb_models)
            run_type = "test_full"
            logger.info("Limited full training completed successfully")

        # Save forecasts to TEST database table
        logger.info(
            f"Saving forecasts to test SQL database ({DatabaseConfig.TEST_FORECAST_TABLE})..."
        )
        db_manager.save_forecasts_to_sql(
            mlb_forecast, DatabaseConfig.TEST_FORECAST_TABLE
        )
        logger.info("Test forecasts saved successfully!")

        # Save trained/updated models for continuous learning (TEST FILES)
        logger.info("Saving trained models for continuous learning...")
        save_models(mlb_models, test_models_file)
        logger.info(f"Saved {len(mlb_models)} trained models to {test_models_file}")

        # Save test forecast data to pickle
        test_forecast_file = (
            AppConfig.MLB_FORECAST_FILE.parent / "test_mlb_forecast.pkl"
        )
        import pickle

        with test_forecast_file.open("wb") as f:
            pickle.dump(mlb_forecast, f)
        logger.info(f"Saved test forecast data to {test_forecast_file}")

        # Calculate execution time and log successful completion
        end_time = time.time()
        execution_time = end_time - start_time

        logger.info("TEST INCREMENTAL PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info(f"Training mode: {run_type}")
        logger.info(f"Execution time: {execution_time:.1f} seconds")
        logger.info(f"Records processed: {records_processed:,}")
        logger.info(f"Models updated: {models_updated}")
        logger.info(
            f"Processed {len(mlb_models)} MLBs out of {AppConfig.TEST_MLB_COUNT} requested"
        )

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
        logger.error(f"Test incremental pipeline failed with error: {error_message}")

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
    logger.info("Running test incremental pipeline (limited to 5 MLBs)")
    main()
