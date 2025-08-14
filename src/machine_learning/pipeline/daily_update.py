"""
Daily update pipeline for continuous learning.
This script performs incremental model updates with new data since the last successful run.
Includes validation and rollback mechanisms for production stability.
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
from database_utils import db_manager
from data_management.clean_sql_data import process_sales_data
from data_management.feature_creation import create_time_series_features
from data_management.metadata_tracker import (
    create_metadata_table,
    log_pipeline_run,
    get_last_successful_run,
)
from data_management.import_SQL import import_data_from_sql_since_date
from data_management.data_merger import get_date_range_info
from estimation.model_forecast import (
    update_mlb_models_incremental,
    validate_model_improvement,
)
from estimation.model_storage import save_models, load_models, archive_models
from validation.forecast_validator import (
    validate_forecasts,
    calculate_historical_stats,
    validate_forecast_trends,
)
from validation.model_validator import (
    validate_model_consistency,
    monitor_memory_usage_during_validation,
)

# Configure plotting backend
pd.options.plotting.backend = "matplotlib"

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def process_mlb_batch_with_validation(
    batch_mlbs: list,
    original_models: Dict,
    feature_data: pd.DataFrame,
    rollback_stats: Dict,
) -> Dict:
    """
    Process a batch of MLBs with incremental updates, validation, and rollback.

    Args:
        batch_mlbs: List of MLB codes to process
        original_models: Dictionary of original models
        feature_data: Full dataset with features
        rollback_stats: Dictionary to track rollback statistics

    Returns:
        Dictionary with 'models' and 'forecasts' keys containing successful updates
    """
    batch_results = {"models": {}, "forecasts": {}}

    # Conservative additional rounds for daily updates
    additional_rounds = 25

    for mlb in batch_mlbs:
        try:
            logger.info(f"Processing MLB {mlb}...")

            # Get data for this specific MLB
            mlb_data = feature_data[feature_data["mlb"] == mlb].copy()
            if len(mlb_data) < 10:  # Need minimum data for meaningful update
                logger.warning(
                    f"MLB {mlb}: Insufficient data ({len(mlb_data)} rows), skipping"
                )
                rollback_stats["mlbs_failed"] += 1
                continue

            # Get original model
            original_model = original_models[mlb]

            # Perform incremental update with conservative additional rounds
            try:
                updated_model = update_mlb_models_incremental(
                    {mlb: original_model}, mlb_data, additional_rounds=additional_rounds
                )[mlb]

                # Validate model improvement
                # Create validation data from the last 30% of MLB data
                val_size = max(
                    10, len(mlb_data) // 3
                )  # At least 10 samples for validation
                val_data = mlb_data.iloc[-val_size:]

                # Prepare validation features and targets for 90-day forecasting
                FEATURES = ["day_of_week", "day_of_month", "rolling_mean_3", "lag_1"]
                forecast_days = AppConfig.FORECAST_DAYS_LONG

                # Create validation data similar to training format
                X_val = []
                y_val = []
                for i in range(len(val_data) - forecast_days):
                    X_val.append(val_data.iloc[i][FEATURES].values)
                    y_val.append(
                        val_data.iloc[i + 1 : i + forecast_days + 1]["quant"].values
                    )

                if len(X_val) > 0:
                    X_val = pd.DataFrame(X_val, columns=FEATURES)
                    y_columns = [f"quant_{i+1}" for i in range(forecast_days)]
                    y_val = pd.DataFrame(y_val, columns=y_columns)

                    # Validate improvement
                    validation_result = validate_model_improvement(
                        original_model, updated_model, (X_val, y_val)
                    )

                    # Decision logic: keep updated model if improved or maintained performance
                    # (within 5% degradation threshold)
                    improvement_threshold = -5.0  # Allow up to 5% degradation

                    if (
                        validation_result["improvement_percentage"]
                        >= improvement_threshold
                    ):
                        if validation_result["improvement_percentage"] > 0:
                            rollback_stats["models_improved"] += 1
                            decision = "improved"
                        else:
                            rollback_stats["models_maintained"] += 1
                            decision = "maintained"

                        batch_results["models"][mlb] = updated_model

                        # Generate forecast for this updated model
                        forecast_result = generate_forecast_for_mlb(
                            mlb, updated_model, mlb_data
                        )
                        if forecast_result:
                            batch_results["forecasts"][mlb] = forecast_result

                        logger.info(
                            f"MLB {mlb}: Model {decision} "
                            f"({validation_result['improvement_percentage']:.2f}% change), keeping update"
                        )
                    else:
                        # Rollback to original model
                        rollback_stats["models_rolled_back"] += 1
                        logger.info(
                            f"MLB {mlb}: Model degraded "
                            f"({validation_result['improvement_percentage']:.2f}% change), rolling back"
                        )
                        # Don't include in batch_results, effectively keeping original model
                else:
                    # Not enough validation data, keep the update but log warning
                    rollback_stats["models_maintained"] += 1
                    batch_results["models"][mlb] = updated_model
                    logger.warning(
                        f"MLB {mlb}: Insufficient validation data, keeping update by default"
                    )

            except Exception as update_error:
                logger.error(f"MLB {mlb}: Incremental update failed: {update_error}")
                rollback_stats["mlbs_failed"] += 1
                continue

        except Exception as e:
            logger.error(f"MLB {mlb}: Processing failed: {e}")
            rollback_stats["mlbs_failed"] += 1
            continue

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


def main():
    """Main execution function for the daily update pipeline."""
    start_time = time.time()
    run_type = "daily"
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
        logger.info("=== DAILY UPDATE PIPELINE STARTED ===")

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
        if not AppConfig.MLB_REGRESSORS_FILE.exists():
            error_message = f"No existing models found at {AppConfig.MLB_REGRESSORS_FILE}. Run full training first."
            logger.error(error_message)
            raise FileNotFoundError(error_message)

        try:
            original_models = load_models(AppConfig.MLB_REGRESSORS_FILE)
            logger.info(f"Loaded {len(original_models)} existing models")
        except Exception as e:
            error_message = f"Failed to load existing models: {e}"
            logger.error(error_message)
            raise

        # Step 4.5: Backup existing models before updating
        logger.info("Step 4.5: Backing up existing models...")
        try:
            archive_dir = AppConfig.MLB_REGRESSORS_FILE.parent / "archive"
            archive_models(AppConfig.MLB_REGRESSORS_FILE, archive_dir)
            logger.info(f"Models backed up to {archive_dir}")
        except Exception as e:
            logger.warning(f"Failed to backup models: {e}. Continuing with updates...")

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

        # Step 5: Process each MLB with validation and rollback mechanism
        logger.info(
            "Step 5: Processing MLBs with incremental updates and validation..."
        )

        # Get MLBs that have new data and existing models
        mlbs_with_new_data = set(feature_data["mlb"].unique())
        mlbs_with_models = set(original_models.keys())
        mlbs_to_update = mlbs_with_new_data.intersection(mlbs_with_models)

        logger.info(f"Found {len(mlbs_with_new_data)} MLBs with new data")
        logger.info(f"Found {len(mlbs_with_models)} MLBs with existing models")
        logger.info(
            f"Will update {len(mlbs_to_update)} MLBs that have both new data and models"
        )

        updated_models = {}
        final_forecasts = {}
        rollback_stats["total_processed"] = len(mlbs_to_update)

        # Process MLBs in batches for memory management
        batch_size = 10  # Conservative batch size
        mlb_list = list(mlbs_to_update)

        for i in range(0, len(mlb_list), batch_size):
            batch_mlbs = mlb_list[i : i + batch_size]
            logger.info(
                f"Processing batch {i//batch_size + 1}: MLBs {i+1}-{min(i+batch_size, len(mlb_list))} of {len(mlb_list)}"
            )

            batch_results = process_mlb_batch_with_validation(
                batch_mlbs, original_models, feature_data, rollback_stats
            )

            updated_models.update(batch_results["models"])
            final_forecasts.update(batch_results["forecasts"])

            # Memory cleanup after each batch
            gc.collect()
            logger.debug(f"Completed batch {i//batch_size + 1}, memory cleaned up")

        models_updated = len(updated_models)

        # Log rollback statistics
        logger.info("=== ROLLBACK STATISTICS ===")
        logger.info(f"Total MLBs processed: {rollback_stats['total_processed']}")
        logger.info(f"Models improved: {rollback_stats['models_improved']}")
        logger.info(f"Models maintained: {rollback_stats['models_maintained']}")
        logger.info(f"Models rolled back: {rollback_stats['models_rolled_back']}")
        logger.info(f"MLBs failed: {rollback_stats['mlbs_failed']}")
        logger.info(f"Final models saved: {models_updated}")

        # Step 5.5: Additional model validation and consistency checks
        if updated_models:
            logger.info("Step 5.5: Performing additional model validation...")
            try:
                is_consistent, consistency_issues = validate_model_consistency(
                    updated_models
                )
                if not is_consistent or consistency_issues:
                    logger.warning(
                        f"Model consistency check found {len(consistency_issues)} issues:"
                    )
                    for issue in consistency_issues[:5]:  # Log first 5 issues
                        logger.warning(f"  - {issue}")
                else:
                    logger.info("All models passed consistency validation")

                # Monitor memory usage during validation
                memory_stats = monitor_memory_usage_during_validation()
                logger.info(
                    f"Memory usage during validation: {memory_stats['rss_mb']:.1f} MB RSS, "
                    f"{memory_stats['percent']:.1f}% of system memory"
                )

            except Exception as e:
                logger.error(f"Additional model validation failed: {e}")
                logger.warning(
                    "Continuing with models that passed initial validation..."
                )

        # Step 6: Validate forecasts before saving
        if final_forecasts:
            logger.info(f"Step 6: Validating {len(final_forecasts)} forecasts...")
            try:
                validated_forecasts, validation_issues = validate_forecasts(
                    final_forecasts, historical_stats
                )

                # Log validation results
                if validation_issues:
                    logger.warning(
                        f"Forecast validation found {len(validation_issues)} issues:"
                    )
                    for issue in validation_issues[:10]:  # Log first 10 issues
                        logger.warning(f"  - {issue}")
                    if len(validation_issues) > 10:
                        logger.warning(
                            f"  ... and {len(validation_issues) - 10} more issues"
                        )

                # Optional: Check forecast trends for additional warnings
                trend_warnings = validate_forecast_trends(
                    validated_forecasts, historical_stats
                )
                if trend_warnings:
                    logger.info(
                        f"Forecast trend analysis found {len(trend_warnings)} warnings:"
                    )
                    for warning in trend_warnings[:5]:  # Log first 5 warnings
                        logger.info(f"  - {warning}")

                # Use validated forecasts for saving
                final_forecasts = validated_forecasts
                logger.info(
                    f"Forecast validation completed: {len(final_forecasts)} forecasts validated"
                )

            except Exception as e:
                logger.error(f"Forecast validation failed: {e}")
                logger.warning("Proceeding with unvalidated forecasts...")

        # Step 6.5: Save forecasts to SQL
        if final_forecasts:
            logger.info(
                f"Step 6.5: Saving {len(final_forecasts)} validated forecasts to SQL database..."
            )
            try:
                db_manager.save_forecasts_to_sql(final_forecasts)
                logger.info("Forecasts saved successfully!")
            except Exception as e:
                logger.error(f"Failed to save forecasts: {e}")
                # Don't fail the entire pipeline for forecast saving issues
        else:
            logger.info(
                "No forecasts to save (no models were updated successfully or all failed validation)"
            )

        # Step 7: Save updated models (only models that improved or maintained performance)
        if updated_models:
            logger.info(f"Step 7: Saving {len(updated_models)} updated models...")
            try:
                save_models(updated_models, AppConfig.MLB_REGRESSORS_FILE)
                logger.info(
                    f"Saved {len(updated_models)} updated models to {AppConfig.MLB_REGRESSORS_FILE}"
                )
            except Exception as e:
                logger.error(f"Failed to save updated models: {e}")
                raise
        else:
            logger.info(
                "No updated models to save (all models were rolled back or failed)"
            )

        logger.info("Daily update pipeline - MLB processing completed successfully")

        # Step 8: Log metadata including rollback statistics
        end_time = time.time()
        execution_time = end_time - start_time

        logger.info("DAILY UPDATE PIPELINE COMPLETED SUCCESSFULLY!")
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

        logger.info(f"Pipeline completed with status: {final_status}")
        if error_message:
            logger.warning(f"Issues encountered: {error_message}")

    except Exception as e:
        error_message = str(e)
        logger.error(f"Daily update pipeline failed with error: {error_message}")

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


def main_with_specific_date(since_date: str):
    """
    Alternative main function that uses a specific date for data import.

    Args:
        since_date: Date string in YYYY-MM-DD format to import data since this date
    """
    start_time = time.time()
    run_type = "daily_specific"
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
        logger.info(f"=== DAILY UPDATE PIPELINE STARTED (since {since_date}) ===")

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

        # Import data since specified date (skip step 1 - last run check)
        logger.info(f"Importing data since {since_date}...")
        try:
            new_data = import_data_from_sql_since_date(
                user=DatabaseConfig.USER,
                password=DatabaseConfig.PASSWORD,
                host=DatabaseConfig.HOST,
                port=DatabaseConfig.PORT,
                dbname=DatabaseConfig.DBNAME,
                view=DatabaseConfig.VIEW,
                since_date=since_date,
            )
            records_processed = len(new_data)
            logger.info(f"Imported {records_processed:,} records since {since_date}")
        except Exception as e:
            error_message = f"Failed to import data since {since_date}: {e}"
            logger.error(error_message)
            raise

        if new_data.empty:
            logger.info(
                f"No new data found since {since_date}. Pipeline completed with no updates."
            )
            log_pipeline_run(
                run_type=run_type,
                status="success",
                records_processed=0,
                models_updated=0,
                run_duration_seconds=time.time() - start_time,
            )
            return

        # Step 4: Load existing models
        logger.info("Step 4: Loading existing models...")
        if not AppConfig.MLB_REGRESSORS_FILE.exists():
            error_message = f"No existing models found at {AppConfig.MLB_REGRESSORS_FILE}. Run full training first."
            logger.error(error_message)
            raise FileNotFoundError(error_message)

        try:
            original_models = load_models(AppConfig.MLB_REGRESSORS_FILE)
            logger.info(f"Loaded {len(original_models)} existing models")
        except Exception as e:
            error_message = f"Failed to load existing models: {e}"
            logger.error(error_message)
            raise

        # Step 4.5: Backup existing models before updating
        logger.info("Step 4.5: Backing up existing models...")
        try:
            archive_dir = AppConfig.MLB_REGRESSORS_FILE.parent / "archive"
            archive_models(AppConfig.MLB_REGRESSORS_FILE, archive_dir)
            logger.info(f"Models backed up to {archive_dir}")
        except Exception as e:
            logger.warning(f"Failed to backup models: {e}. Continuing with updates...")

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

        # Step 5: Process each MLB with validation and rollback mechanism
        logger.info(
            "Step 5: Processing MLBs with incremental updates and validation..."
        )

        # Get MLBs that have new data and existing models
        mlbs_with_new_data = set(feature_data["mlb"].unique())
        mlbs_with_models = set(original_models.keys())
        mlbs_to_update = mlbs_with_new_data.intersection(mlbs_with_models)

        logger.info(f"Found {len(mlbs_with_new_data)} MLBs with new data")
        logger.info(f"Found {len(mlbs_with_models)} MLBs with existing models")
        logger.info(
            f"Will update {len(mlbs_to_update)} MLBs that have both new data and models"
        )

        updated_models = {}
        final_forecasts = {}
        rollback_stats["total_processed"] = len(mlbs_to_update)

        # Process MLBs in batches for memory management
        batch_size = 10  # Conservative batch size
        mlb_list = list(mlbs_to_update)

        for i in range(0, len(mlb_list), batch_size):
            batch_mlbs = mlb_list[i : i + batch_size]
            logger.info(
                f"Processing batch {i//batch_size + 1}: MLBs {i+1}-{min(i+batch_size, len(mlb_list))} of {len(mlb_list)}"
            )

            batch_results = process_mlb_batch_with_validation(
                batch_mlbs, original_models, feature_data, rollback_stats
            )

            updated_models.update(batch_results["models"])
            final_forecasts.update(batch_results["forecasts"])

            # Memory cleanup after each batch
            gc.collect()
            logger.debug(f"Completed batch {i//batch_size + 1}, memory cleaned up")

        models_updated = len(updated_models)

        # Log rollback statistics
        logger.info("=== ROLLBACK STATISTICS ===")
        logger.info(f"Total MLBs processed: {rollback_stats['total_processed']}")
        logger.info(f"Models improved: {rollback_stats['models_improved']}")
        logger.info(f"Models maintained: {rollback_stats['models_maintained']}")
        logger.info(f"Models rolled back: {rollback_stats['models_rolled_back']}")
        logger.info(f"MLBs failed: {rollback_stats['mlbs_failed']}")
        logger.info(f"Final models saved: {models_updated}")

        # Step 5.5: Additional model validation and consistency checks
        if updated_models:
            logger.info("Step 5.5: Performing additional model validation...")
            try:
                is_consistent, consistency_issues = validate_model_consistency(
                    updated_models
                )
                if not is_consistent or consistency_issues:
                    logger.warning(
                        f"Model consistency check found {len(consistency_issues)} issues:"
                    )
                    for issue in consistency_issues[:5]:  # Log first 5 issues
                        logger.warning(f"  - {issue}")
                else:
                    logger.info("All models passed consistency validation")

                # Monitor memory usage during validation
                memory_stats = monitor_memory_usage_during_validation()
                logger.info(
                    f"Memory usage during validation: {memory_stats['rss_mb']:.1f} MB RSS, "
                    f"{memory_stats['percent']:.1f}% of system memory"
                )

            except Exception as e:
                logger.error(f"Additional model validation failed: {e}")
                logger.warning(
                    "Continuing with models that passed initial validation..."
                )

        # Step 6: Validate forecasts before saving
        if final_forecasts:
            logger.info(f"Step 6: Validating {len(final_forecasts)} forecasts...")
            try:
                validated_forecasts, validation_issues = validate_forecasts(
                    final_forecasts, historical_stats
                )

                # Log validation results
                if validation_issues:
                    logger.warning(
                        f"Forecast validation found {len(validation_issues)} issues:"
                    )
                    for issue in validation_issues[:10]:  # Log first 10 issues
                        logger.warning(f"  - {issue}")
                    if len(validation_issues) > 10:
                        logger.warning(
                            f"  ... and {len(validation_issues) - 10} more issues"
                        )

                # Optional: Check forecast trends for additional warnings
                trend_warnings = validate_forecast_trends(
                    validated_forecasts, historical_stats
                )
                if trend_warnings:
                    logger.info(
                        f"Forecast trend analysis found {len(trend_warnings)} warnings:"
                    )
                    for warning in trend_warnings[:5]:  # Log first 5 warnings
                        logger.info(f"  - {warning}")

                # Use validated forecasts for saving
                final_forecasts = validated_forecasts
                logger.info(
                    f"Forecast validation completed: {len(final_forecasts)} forecasts validated"
                )

            except Exception as e:
                logger.error(f"Forecast validation failed: {e}")
                logger.warning("Proceeding with unvalidated forecasts...")

        # Step 6.5: Save forecasts to SQL
        if final_forecasts:
            logger.info(
                f"Step 6.5: Saving {len(final_forecasts)} validated forecasts to SQL database..."
            )
            try:
                db_manager.save_forecasts_to_sql(final_forecasts)
                logger.info("Forecasts saved successfully!")
            except Exception as e:
                logger.error(f"Failed to save forecasts: {e}")
                # Don't fail the entire pipeline for forecast saving issues
        else:
            logger.info(
                "No forecasts to save (no models were updated successfully or all failed validation)"
            )

        # Step 7: Save updated models (only models that improved or maintained performance)
        if updated_models:
            logger.info(f"Step 7: Saving {len(updated_models)} updated models...")
            try:
                save_models(updated_models, AppConfig.MLB_REGRESSORS_FILE)
                logger.info(
                    f"Saved {len(updated_models)} updated models to {AppConfig.MLB_REGRESSORS_FILE}"
                )
            except Exception as e:
                logger.error(f"Failed to save updated models: {e}")
                raise
        else:
            logger.info(
                "No updated models to save (all models were rolled back or failed)"
            )

        logger.info("Daily update pipeline - MLB processing completed successfully")

        # Step 8: Log metadata including rollback statistics
        end_time = time.time()
        execution_time = end_time - start_time

        logger.info("DAILY UPDATE PIPELINE COMPLETED SUCCESSFULLY!")
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

        logger.info(f"Pipeline completed with status: {final_status}")
        if error_message:
            logger.warning(f"Issues encountered: {error_message}")

    except Exception as e:
        error_message = str(e)
        logger.error(
            f"Daily update pipeline with specific date failed: {error_message}"
        )
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
        db_manager.close_connection()


if __name__ == "__main__":
    import sys

    # Check if specific date argument is provided
    if len(sys.argv) > 1:
        specific_date = sys.argv[1]
        logger.info(
            f"Running daily update pipeline with specific date: {specific_date}"
        )
        main_with_specific_date(specific_date)
    else:
        logger.info("Running daily update pipeline")
        main()
