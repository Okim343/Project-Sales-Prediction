"""
Test version of the unified pipeline runner for continuous learning.

This script mirrors pipeline_runner.py exactly but with testing limitations:
- Limited to a small number of MLBs for testing purposes to avoid large file size issues
- Uses limited forecasting functions where available
- All other functionality remains identical to the main script
"""

import gc
import logging
import numpy as np
import pandas as pd
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Tuple, Optional

# Add src path to import the modules
sys.path.append(str(Path(__file__).parent.parent))

from config import AppConfig, DatabaseConfig
from database_utils import db_manager, validate_data_freshness
from data_management.clean_sql_data import process_sales_data
from data_management.feature_creation import create_time_series_features
from data_management.metadata_tracker import (
    create_metadata_table,
    log_pipeline_run,
    get_last_successful_run,
)
from data_management.import_SQL import import_data_from_sql_since_date
from data_management.data_merger import (
    merge_with_historical,
    get_date_range_info,
)
from estimation.model_forecast import (
    update_mlb_models_incremental,
    forecast_future_sales_direct_limited,  # Use limited version for testing
)
from estimation.model_storage import save_models, load_models, archive_models
from validation.forecast_validator import (
    calculate_historical_stats,
)
from validation.model_validator import (
    monitor_memory_usage_during_validation,
)

# Import integrated validation module
from pipeline.integrated_validator import (
    ValidationContext,
    process_mlb_batch_integrated,
    validate_final_results,
)

# Configure plotting backend
pd.options.plotting.backend = "matplotlib"

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Testing constants
MAX_MLBS_FOR_TESTING = 5


def process_mlb_batch_with_validation_limited(
    batch_mlbs: list,
    original_models: Dict,
    feature_data: pd.DataFrame,
    rollback_stats: Dict,
    max_mlbs: int = MAX_MLBS_FOR_TESTING,
) -> Dict:
    """
    Limited version of MLB batch processing for testing using integrated validation.
    Process up to max_mlbs MLBs with incremental updates, validation, and rollback.

    Args:
        batch_mlbs: List of MLB codes to process
        original_models: Dictionary of original models
        feature_data: Full dataset with features
        rollback_stats: Dictionary to track rollback statistics
        max_mlbs: Maximum number of MLBs to process (for testing)

    Returns:
        Dictionary with 'models' and 'forecasts' keys containing successful updates
    """
    logger.info(f"Using integrated validation for {max_mlbs} MLBs in test mode")

    # Create validation context for integrated approach with testing constraints
    historical_stats = calculate_historical_stats(feature_data)
    context = ValidationContext(
        historical_stats=historical_stats,
        rollback_stats=rollback_stats,
        improvement_threshold=-5.0,  # Allow up to 5% degradation
        additional_rounds=25,  # Conservative rounds for daily updates
    )

    # Limit batch to testing constraints
    limited_batch_mlbs = batch_mlbs[:max_mlbs]
    if len(limited_batch_mlbs) < len(batch_mlbs):
        logger.info(
            f"Limited batch from {len(batch_mlbs)} to {len(limited_batch_mlbs)} MLBs for testing"
        )

    # Use the integrated validation approach
    batch_results = process_mlb_batch_integrated(
        limited_batch_mlbs, original_models, feature_data, context
    )

    logger.info(
        f"Integrated validation completed for {len(batch_results['models'])} MLBs in test mode"
    )
    return batch_results


def generate_forecast_for_mlb(
    mlb: str, model, mlb_data: pd.DataFrame
) -> Optional[Tuple]:
    """
    Generate forecast for a single MLB using its updated model.

    Args:
        mlb: MLB code
        model: Trained model
        mlb_data: Data for this MLB

    Returns:
        Tuple of (forecast_df, sku) or None if failed
    """
    try:
        FEATURES = ["day_of_week", "day_of_month", "rolling_mean_3", "lag_1"]
        forecast_days = AppConfig.FORECAST_DAYS_LONG

        # Get SKU for this MLB
        sku = mlb_data["sku"].iloc[0] if "sku" in mlb_data.columns else None

        # Use the last row of features for prediction
        last_features = mlb_data.iloc[-1][FEATURES].values.reshape(1, -1)

        # Generate predictions
        if hasattr(model, "estimators_"):
            # MultiOutputRegressor
            predictions = model.predict(last_features)[0]
        else:
            # Regular model
            predictions = model.predict(last_features)
            if len(predictions.shape) == 1:
                predictions = predictions.reshape(1, -1)[0]

        # Ensure correct number of predictions
        if len(predictions) != forecast_days:
            if len(predictions) < forecast_days:
                last_pred = predictions[-1] if len(predictions) > 0 else 0
                predictions = list(predictions) + [last_pred] * (
                    forecast_days - len(predictions)
                )
            else:
                predictions = predictions[:forecast_days]

        # Round predictions to nearest integers (products sold in whole units)
        predictions = np.round(predictions).astype(int)

        # Create forecast DataFrame
        today = pd.Timestamp.now().normalize()
        future_start = today + pd.Timedelta(days=1)
        future_dates = pd.date_range(
            start=future_start, periods=forecast_days, freq="D"
        )

        forecast_df = pd.DataFrame({"prediction": predictions}, index=future_dates)
        return (forecast_df, sku)

    except Exception as e:
        logger.error(f"Failed to generate forecast for MLB {mlb}: {e}")
        return None


def generate_forecasts_for_existing_models_limited(
    updated_models: dict, feature_data: pd.DataFrame, forecast_days: int
) -> tuple[dict, dict]:
    """
    Limited version: Generate forecasts only for MLBs that have existing/updated models.
    This avoids iterating through the entire dataset when we only want to forecast
    for specific MLBs we have models for.

    Args:
        updated_models: Dictionary of MLB -> trained model
        feature_data: Full dataset with all MLBs
        forecast_days: Number of days to forecast

    Returns:
        tuple[dict, dict]: (mlb_forecasts, mlb_models) same format as other forecast functions
    """
    mlb_forecasts = {}
    mlb_models = {}

    logger.info(
        f"Generating targeted forecasts for {len(updated_models)} specific MLBs (test mode)"
    )

    processed_count = 0
    for mlb, model in updated_models.items():
        if processed_count >= MAX_MLBS_FOR_TESTING:
            logger.info(f"Reached test limit of {MAX_MLBS_FOR_TESTING} MLBs")
            break

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
            processed_count += 1

        except Exception as e:
            logger.error(f"Failed to generate forecast for MLB {mlb}: {e}")
            continue

    logger.info(
        f"Successfully generated forecasts for {len(mlb_forecasts)} MLBs (test mode)"
    )
    return mlb_forecasts, mlb_models


def run_daily_mode():
    """Execute daily mode: incremental updates since last successful run."""
    start_time = time.time()
    run_type = "daily_test"
    records_processed = 0
    models_updated = 0
    error_message = None
    rollback_stats = {
        "models_improved": 0,
        "models_maintained": 0,
        "models_rolled_back": 0,
        "mlbs_failed": 0,
        "total_processed": 0,
    }

    try:
        logger.info("=== DAILY MODE PIPELINE STARTED (TEST VERSION) ===")

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

        # Step 1: Check last successful run date from metadata
        logger.info("Step 1: Checking last successful run date...")
        last_run = get_last_successful_run(run_type="daily")

        if last_run:
            since_date = last_run["run_timestamp"].date()
            logger.info(f"Last successful daily run: {since_date}")
        else:
            # If no previous daily runs, look for any successful run as fallback
            last_run = get_last_successful_run()
            if last_run:
                since_date = last_run["run_timestamp"].date()
                logger.info(
                    f"No daily runs found, using last successful run: {since_date}"
                )
            else:
                # Default to 7 days ago if no runs found
                since_date = (datetime.now() - timedelta(days=7)).date()
                logger.info(
                    f"No previous runs found, defaulting to 7 days ago: {since_date}"
                )

        # Step 2: Import data since that date
        logger.info(f"Step 2: Importing data since {since_date}...")
        try:
            new_data = import_data_from_sql_since_date(
                user=DatabaseConfig.USER,
                password=DatabaseConfig.PASSWORD,
                host=DatabaseConfig.HOST,
                port=DatabaseConfig.PORT,
                dbname=DatabaseConfig.DBNAME,
                view=DatabaseConfig.VIEW,
                since_date=str(since_date),
            )
            records_processed = len(new_data)
            logger.info(
                f"Imported {records_processed:,} new records since {since_date}"
            )
        except Exception as e:
            error_message = f"Failed to import data since {since_date}: {e}"
            logger.error(error_message)
            raise

        # Step 3: If no new data, log and exit gracefully
        if new_data.empty:
            logger.info(
                f"No new data found since {since_date}. Pipeline completed with no updates."
            )
            # Log successful but no-op run
            end_time = time.time()
            execution_time = end_time - start_time
            log_pipeline_run(
                run_type=run_type,
                status="success",
                records_processed=0,
                models_updated=0,
                run_duration_seconds=execution_time,
            )
            return

        # Step 4: Load existing models
        logger.info("Step 4: Loading existing models...")
        models_file = AppConfig.MLB_REGRESSORS_FILE.parent / "test_mlb_regressors.pkl"
        if not models_file.exists():
            error_message = f"No existing test models found at {models_file}. Run full test training first."
            logger.error(error_message)
            raise FileNotFoundError(error_message)

        try:
            original_models = load_models(models_file)
            logger.info(f"Loaded {len(original_models)} existing test models")
        except Exception as e:
            error_message = f"Failed to load existing test models: {e}"
            logger.error(error_message)
            raise

        # Step 4.5: Backup existing models before updating
        logger.info("Step 4.5: Backing up existing models...")
        try:
            archive_dir = models_file.parent / "archive"
            archive_models(models_file, archive_dir)
            logger.info(f"Test models backed up to {archive_dir}")
        except Exception as e:
            logger.warning(
                f"Failed to backup test models: {e}. Continuing with updates..."
            )

        # Process and clean the new data
        logger.info("Processing and cleaning new data...")
        clean_data = process_sales_data(new_data)
        feature_data = create_time_series_features(clean_data)

        # Calculate historical stats for forecast validation
        logger.info("Calculating historical statistics for validation...")
        historical_stats = calculate_historical_stats(feature_data)

        # Get date range info for logging
        date_info = get_date_range_info(feature_data)
        if date_info["min_date"] and date_info["max_date"]:
            logger.info(
                f"New data spans from {date_info['min_date']} to {date_info['max_date']} "
                f"({date_info['date_span_days']} days, {date_info['record_count']:,} records)"
            )

        # Step 5: Process each MLB with validation and rollback mechanism (LIMITED FOR TESTING)
        logger.info(
            f"Step 5: Processing MLBs with incremental updates and validation (limited to {MAX_MLBS_FOR_TESTING} MLBs)..."
        )

        # Get MLBs that have new data and existing models
        mlbs_with_new_data = set(feature_data["mlb"].unique())
        mlbs_with_models = set(original_models.keys())
        mlbs_to_update = mlbs_with_new_data.intersection(mlbs_with_models)

        logger.info(f"Found {len(mlbs_with_new_data)} MLBs with new data")
        logger.info(f"Found {len(mlbs_with_models)} MLBs with existing models")
        logger.info(
            f"Will update {min(len(mlbs_to_update), MAX_MLBS_FOR_TESTING)} MLBs (limited for testing)"
        )

        updated_models = {}
        final_forecasts = {}
        rollback_stats["total_processed"] = min(
            len(mlbs_to_update), MAX_MLBS_FOR_TESTING
        )

        # Process MLBs in batches for memory management (limited for testing)
        batch_size = 5  # Smaller batch size for testing
        mlb_list = list(mlbs_to_update)[:MAX_MLBS_FOR_TESTING]

        for i in range(0, len(mlb_list), batch_size):
            batch_mlbs = mlb_list[i : i + batch_size]
            logger.info(
                f"Processing test batch {i//batch_size + 1}: MLBs {i+1}-{min(i+batch_size, len(mlb_list))} of {len(mlb_list)}"
            )

            batch_results = process_mlb_batch_with_validation_limited(
                batch_mlbs, original_models, feature_data, rollback_stats
            )

            updated_models.update(batch_results["models"])
            final_forecasts.update(batch_results["forecasts"])

            # Memory cleanup after each batch
            gc.collect()
            logger.debug(f"Completed test batch {i//batch_size + 1}, memory cleaned up")

        models_updated = len(updated_models)

        # Log rollback statistics
        logger.info("=== ROLLBACK STATISTICS (TEST MODE) ===")
        logger.info(f"Total MLBs processed: {rollback_stats['total_processed']}")
        logger.info(f"Models improved: {rollback_stats['models_improved']}")
        logger.info(f"Models maintained: {rollback_stats['models_maintained']}")
        logger.info(f"Models rolled back: {rollback_stats['models_rolled_back']}")
        logger.info(f"MLBs failed: {rollback_stats['mlbs_failed']}")
        logger.info(f"Final models saved: {models_updated}")

        # Step 5.5: Integrated final validation (test mode - replaces separate model and forecast validation)
        if updated_models or final_forecasts:
            logger.info(
                "Step 5.5: Performing integrated final validation (test mode)..."
            )
            try:
                # Create validation context for final check
                context = ValidationContext(
                    historical_stats=historical_stats, rollback_stats=rollback_stats
                )

                # Use integrated final validation
                validated_models, validated_forecasts, validation_issues = (
                    validate_final_results(updated_models, final_forecasts, context)
                )

                # Log validation results
                if validation_issues:
                    logger.warning(
                        f"Final validation found {len(validation_issues)} issues (test mode):"
                    )
                    for issue in validation_issues[:10]:  # Log first 10 issues
                        logger.warning(f"  - {issue}")
                    if len(validation_issues) > 10:
                        logger.warning(
                            f"  ... and {len(validation_issues) - 10} more issues"
                        )
                else:
                    logger.info(
                        "All models and forecasts passed final validation (test mode)"
                    )

                # Use validated results
                updated_models = validated_models
                final_forecasts = validated_forecasts

                # Monitor memory usage during validation
                memory_stats = monitor_memory_usage_during_validation()
                logger.info(
                    f"Memory usage during validation: {memory_stats['rss_mb']:.1f} MB RSS, "
                    f"{memory_stats['percent']:.1f}% of system memory"
                )

                logger.info(
                    f"Integrated validation completed (test mode): {len(validated_models)} models, "
                    f"{len(validated_forecasts)} forecasts validated"
                )

            except Exception as e:
                logger.error(f"Integrated final validation failed (test mode): {e}")
                logger.warning("Proceeding with pre-validation results...")

        # Step 6: Save forecasts to SQL (SKIP IN TEST MODE)
        if final_forecasts:
            logger.info(
                f"Step 6: Would save {len(final_forecasts)} forecasts to SQL (SKIPPED in test mode)"
            )
        else:
            logger.info(
                "No forecasts to save (no models were updated successfully or all failed validation)"
            )

        # Step 7: Save updated models (only models that improved or maintained performance)
        if updated_models:
            logger.info(f"Step 7: Saving {len(updated_models)} updated test models...")
            try:
                save_models(updated_models, models_file)
                logger.info(
                    f"Saved {len(updated_models)} updated test models to {models_file}"
                )
            except Exception as e:
                logger.error(f"Failed to save updated test models: {e}")
                raise
        else:
            logger.info(
                "No updated models to save (all models were rolled back or failed)"
            )

        logger.info("Daily mode test pipeline - MLB processing completed successfully")

        # Step 8: Log metadata including rollback statistics
        end_time = time.time()
        execution_time = end_time - start_time

        logger.info("DAILY MODE TEST PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info(f"Execution time: {execution_time:.1f} seconds")
        logger.info(f"Records processed: {records_processed:,}")
        logger.info(f"Models updated: {models_updated}")
        logger.info(
            f"Success rate: {(models_updated / max(rollback_stats['total_processed'], 1) * 100):.1f}%"
        )

        # Determine overall status based on results
        if models_updated == 0:
            final_status = (
                "failed" if rollback_stats["total_processed"] > 0 else "success"
            )
            error_message = (
                "No models were successfully updated"
                if rollback_stats["total_processed"] > 0
                else None
            )
        elif (
            rollback_stats["mlbs_failed"] > 0
            or rollback_stats["models_rolled_back"] > 0
        ):
            final_status = "partial"
        else:
            final_status = "success"

        # Create detailed error message for partial/failed runs
        if final_status in ["partial", "failed"]:
            error_details = []
            if rollback_stats["models_rolled_back"] > 0:
                error_details.append(
                    f"{rollback_stats['models_rolled_back']} models rolled back due to performance degradation"
                )
            if rollback_stats["mlbs_failed"] > 0:
                error_details.append(
                    f"{rollback_stats['mlbs_failed']} MLBs failed to process"
                )
            if error_details:
                error_message = "; ".join(error_details)

        # Log pipeline run with enhanced metadata
        log_pipeline_run(
            run_type=run_type,
            status=final_status,
            records_processed=records_processed,
            models_updated=models_updated,
            error_message=error_message,
            run_duration_seconds=execution_time,
        )

        logger.info(f"Test pipeline completed with status: {final_status}")
        if error_message:
            logger.warning(f"Issues encountered: {error_message}")

    except Exception as e:
        error_message = str(e)
        logger.error(f"Daily mode test pipeline failed with error: {error_message}")

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


def run_full_mode():
    """Execute full mode: complete training from scratch with fallback capability."""
    start_time = time.time()
    run_type = "full_test"
    records_processed = 0
    models_updated = 0
    error_message = None
    use_incremental = False

    try:
        logger.info("=== FULL MODE PIPELINE STARTED (TEST VERSION) ===")

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
        models_file = AppConfig.MLB_REGRESSORS_FILE.parent / "test_mlb_regressors.pkl"
        if models_file.exists():
            try:
                logger.info(f"Loading existing test models from {models_file}")
                existing_models = load_models(models_file)
                logger.info(f"Loaded {len(existing_models)} existing test models")
                use_incremental = True
            except Exception as e:
                logger.warning(f"Failed to load existing test models: {e}")
                logger.info("Proceeding with full training instead")
                use_incremental = False
        else:
            logger.info("No existing test models found, will perform full training")
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
            logger.info("=== INCREMENTAL TRAINING MODE (TEST) ===")
            logger.info(
                f"Updating {len(existing_models)} existing test models with new data"
            )

            try:
                # Backup existing models before updating
                archive_dir = models_file.parent / "archive"
                archive_models(models_file, archive_dir)

                # Perform incremental model updates (LIMITED FOR TESTING)
                logger.info("Starting incremental model updates (test mode)...")
                # Only update first few models for testing
                limited_models = dict(
                    list(existing_models.items())[:MAX_MLBS_FOR_TESTING]
                )
                updated_models = update_mlb_models_incremental(
                    limited_models,
                    feature_data,
                    additional_rounds=50,  # Conservative number for incremental updates
                )
                models_updated = len(updated_models)
                logger.info(
                    f"Incremental update completed for {models_updated} test models"
                )

                # Generate forecasts using targeted approach for updated models only
                logger.info(
                    f"Generating {AppConfig.FORECAST_DAYS_LONG}-day forecasts for {len(updated_models)} updated test models..."
                )
                mlb_forecast, mlb_models = (
                    generate_forecasts_for_existing_models_limited(
                        updated_models, feature_data, AppConfig.FORECAST_DAYS_LONG
                    )
                )
                logger.info("Targeted forecasting with updated test models complete!")

                logger.info("Successfully completed incremental training test pipeline")

            except Exception as incremental_error:
                logger.error(f"Incremental training test failed: {incremental_error}")
                logger.info("Falling back to limited full training...")

                # Fallback to full training (LIMITED FOR TESTING)
                mlb_forecast, mlb_models = forecast_future_sales_direct_limited(
                    feature_data,
                    AppConfig.FORECAST_DAYS_LONG,
                    max_mlbs=MAX_MLBS_FOR_TESTING,
                )
                models_updated = len(mlb_models)
                run_type = "full_fallback_test"
                logger.info("Fallback full training (test mode) completed successfully")

        else:
            logger.info("=== FULL TRAINING MODE (TEST) ===")
            logger.info("Training models from scratch (limited for testing)")

            # Generate forecasts and train models from scratch (LIMITED FOR TESTING)
            logger.info(
                f"Generating {AppConfig.FORECAST_DAYS_LONG}-day forecasts (test mode)..."
            )
            mlb_forecast, mlb_models = forecast_future_sales_direct_limited(
                feature_data,
                AppConfig.FORECAST_DAYS_LONG,
                max_mlbs=MAX_MLBS_FOR_TESTING,
            )
            models_updated = len(mlb_models)
            run_type = "full_test"
            logger.info("Full training (test mode) completed successfully")

        # Save forecasts to database (SKIP IN TEST MODE)
        logger.info(
            f"Would save {len(mlb_forecast)} forecasts to SQL (SKIPPED in test mode)"
        )

        # Save trained/updated models for continuous learning
        logger.info("Saving trained test models for continuous learning...")
        save_models(mlb_models, models_file)
        logger.info(f"Saved {len(mlb_models)} trained test models to {models_file}")

        # Calculate execution time and log successful completion
        end_time = time.time()
        execution_time = end_time - start_time

        logger.info("FULL MODE TEST PIPELINE COMPLETED SUCCESSFULLY!")
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
        logger.error(f"Full mode test pipeline failed with error: {error_message}")

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


def run_since_date_mode(since_date: str):
    """Execute since-date mode: incremental updates since specified date."""
    start_time = time.time()
    run_type = "incremental_filtered_test"
    records_processed = 0
    models_updated = 0
    error_message = None

    try:
        logger.info(
            f"=== SINCE-DATE MODE PIPELINE STARTED (TEST VERSION, since {since_date}) ==="
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
        models_file = AppConfig.MLB_REGRESSORS_FILE.parent / "test_mlb_regressors.pkl"
        if not models_file.exists():
            error_message = f"No existing test models found at {models_file}. Run full test training first."
            logger.error(error_message)
            raise FileNotFoundError(error_message)

        logger.info(f"Loading existing test models from {models_file}")
        existing_models = load_models(models_file)
        logger.info(f"Loaded {len(existing_models)} existing test models")

        # Import new data since specified date using date-filtered import
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
                "No new data found since the specified date. Test pipeline completed with no updates."
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

        # Perform incremental model updates with new data only (LIMITED FOR TESTING)
        logger.info("Starting incremental model updates with new data (test mode)...")

        # Backup existing models before updating
        archive_dir = models_file.parent / "archive"
        archive_models(models_file, archive_dir)

        # Filter feature_data to only include MLBs that have new data
        mlbs_with_new_data = new_data["mlb"].unique()
        incremental_data = feature_data[feature_data["mlb"].isin(mlbs_with_new_data)]

        # Limit to first few MLBs for testing
        limited_mlbs = mlbs_with_new_data[:MAX_MLBS_FOR_TESTING]
        incremental_data = feature_data[feature_data["mlb"].isin(limited_mlbs)]

        logger.info(
            f"Performing incremental updates for {len(limited_mlbs)} MLBs with new data (limited for testing)"
        )

        # Only update models that exist in our test set
        models_to_update = {
            mlb: existing_models[mlb] for mlb in limited_mlbs if mlb in existing_models
        }

        updated_models = update_mlb_models_incremental(
            models_to_update, incremental_data, additional_rounds=50
        )
        models_updated = len(updated_models)

        # Generate forecasts for updated models
        logger.info(
            f"Generating {AppConfig.FORECAST_DAYS_LONG}-day forecasts for updated test models..."
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

        logger.info(f"Generated forecasts for {len(mlb_forecast)} test MLBs")

        # Save forecasts to database (SKIP IN TEST MODE)
        if mlb_forecast:
            logger.info(
                f"Would save {len(mlb_forecast)} forecasts to SQL (SKIPPED in test mode)"
            )

        # Save updated models
        logger.info("Saving updated test models for continuous learning...")
        save_models(updated_models, models_file)
        logger.info(f"Saved {len(updated_models)} updated test models to {models_file}")

        # Update historical data file with new data (SKIP IN TEST MODE)
        logger.info(
            f"Would update historical data file with {records_processed:,} new records (SKIPPED in test mode)"
        )

        # Calculate execution time and log successful completion
        end_time = time.time()
        execution_time = end_time - start_time

        logger.info("SINCE-DATE MODE TEST PIPELINE COMPLETED SUCCESSFULLY!")
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
        logger.error(f"Since-date mode test pipeline failed: {error_message}")

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


def main():
    """Main function with unified CLI interface for all pipeline test modes."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Unified test pipeline runner for continuous learning with multiple execution modes"
    )
    parser.add_argument(
        "--mode",
        choices=["daily", "full"],
        default="daily",
        help="Pipeline execution mode (default: daily)",
    )
    parser.add_argument(
        "--since-date",
        type=str,
        help="Run incremental updates since specific date (YYYY-MM-DD format)",
    )
    parser.add_argument(
        "since_date_positional",
        nargs="?",
        help="Alternative way to specify since-date for backward compatibility",
    )

    args = parser.parse_args()

    # Handle backward compatibility for positional date argument
    since_date = args.since_date or args.since_date_positional

    if since_date:
        logger.info(f"Running since-date test mode with date: {since_date}")
        run_since_date_mode(since_date)
    elif args.mode == "daily":
        logger.info("Running daily test mode pipeline")
        run_daily_mode()
    elif args.mode == "full":
        logger.info("Running full test mode pipeline")
        run_full_mode()
    else:
        logger.error(f"Unknown mode: {args.mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
