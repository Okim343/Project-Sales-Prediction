"""
Unified pipeline runner for continuous learning with multiple execution modes.

This script consolidates all pipeline functionality into a single entry point with three modes:
1. Daily mode: Incremental updates since last successful run with validation/rollback
2. Full mode: Complete training from scratch (fallback available)
3. Since-date mode: Incremental updates since specified date with data merging

Includes comprehensive validation, rollback mechanisms, and production-ready error handling.
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
from data_management.import_SQL import (
    import_data_from_sql_since_date,
    import_data_last_n_months,
)
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
    validate_monthly_retraining_results,
)

# Import logging utilities for cleaner, more readable logging
from pipeline.logging_utils import (
    log_pipeline_start,
    log_data_info,
    log_model_operations,
    log_rollback_summary,
    log_validation_summary,
    log_pipeline_completion,
    log_memory_usage,
    log_batch_progress,
    log_database_operation,
    log_model_comparison_results,
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
    DEPRECATED: Use process_mlb_batch_integrated instead.

    This function is kept for backward compatibility but now uses the
    integrated validation approach internally.

    Args:
        batch_mlbs: List of MLB codes to process
        original_models: Dictionary of original models
        feature_data: Full dataset with features
        rollback_stats: Dictionary to track rollback statistics

    Returns:
        Dictionary with 'models' and 'forecasts' keys containing successful updates
    """
    # Create validation context for integrated approach
    historical_stats = calculate_historical_stats(feature_data)
    context = ValidationContext(
        historical_stats=historical_stats,
        rollback_stats=rollback_stats,
        improvement_threshold=-5.0,  # Allow up to 5% degradation
        additional_rounds=25,  # Conservative rounds for daily updates
    )

    # Use the integrated validation approach
    return process_mlb_batch_integrated(
        batch_mlbs, original_models, feature_data, context
    )


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


def run_daily_mode():
    """Execute daily mode: incremental updates since last successful run."""
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
        log_pipeline_start("daily", logger)

        # Initialize infrastructure
        logger.debug("Initializing metadata tracking...")
        if not create_metadata_table():
            logger.warning(
                "Failed to create metadata table, continuing without metadata tracking"
            )

        if not db_manager.test_connection():
            error_message = "Database connection failed"
            logger.error(error_message)
            return

        # Determine incremental update baseline
        last_run = get_last_successful_run(run_type="daily")

        if last_run:
            since_date = last_run["run_timestamp"].date()
            logger.info(f"Incremental update since last daily run: {since_date}")
        else:
            # If no previous daily runs, look for any successful run as fallback
            last_run = get_last_successful_run()
            if last_run:
                since_date = last_run["run_timestamp"].date()
                logger.info(
                    f"No daily runs found, updating since last run: {since_date}"
                )
            else:
                # Default to 7 days ago if no runs found
                since_date = (datetime.now() - timedelta(days=7)).date()
                logger.info(
                    f"No previous runs found, updating from 7 days ago: {since_date}"
                )

        # Import incremental data
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
            log_data_info(
                records_processed, logger, operation="imported since " + str(since_date)
            )
        except Exception as e:
            error_message = f"Failed to import data since {since_date}: {e}"
            logger.error(error_message)
            raise

        # Check if incremental update is needed
        if new_data.empty:
            logger.info(
                f"No new data since {since_date} - pipeline completed with no updates"
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

        # Load existing models for incremental updates
        if not AppConfig.MLB_REGRESSORS_FILE.exists():
            error_message = f"No existing models found at {AppConfig.MLB_REGRESSORS_FILE}. Run full training first."
            logger.error(error_message)
            raise FileNotFoundError(error_message)

        try:
            original_models = load_models(AppConfig.MLB_REGRESSORS_FILE)
            log_model_operations("loaded", len(original_models), logger)
        except Exception as e:
            error_message = f"Failed to load existing models: {e}"
            logger.error(error_message)
            raise

        # Backup existing models before incremental updates
        try:
            archive_dir = AppConfig.MLB_REGRESSORS_FILE.parent / "archive"
            archive_models(AppConfig.MLB_REGRESSORS_FILE, archive_dir)
            logger.debug(f"Models backed up to {archive_dir}")
        except Exception as e:
            logger.warning(f"Failed to backup models: {e}. Continuing with updates...")

        # Process new data for model updates
        logger.debug("Processing and cleaning new data...")
        clean_data = process_sales_data(new_data)
        feature_data = create_time_series_features(clean_data)
        historical_stats = calculate_historical_stats(feature_data)

        # Log processed data info
        date_info = get_date_range_info(feature_data)
        if date_info["min_date"] and date_info["max_date"]:
            log_data_info(date_info["record_count"], logger, date_info, "processed")

        # Process MLBs with incremental updates and validation
        mlbs_with_new_data = set(feature_data["mlb"].unique())
        mlbs_with_models = set(original_models.keys())
        mlbs_to_update = mlbs_with_new_data.intersection(mlbs_with_models)

        logger.info(
            f"MLBs for incremental update: {len(mlbs_to_update)} "
            f"(from {len(mlbs_with_new_data)} with new data, {len(mlbs_with_models)} with models)"
        )

        updated_models = {}
        final_forecasts = {}
        rollback_stats["total_processed"] = len(mlbs_to_update)

        # Process MLBs in batches for memory management
        batch_size = 10  # Conservative batch size
        mlb_list = list(mlbs_to_update)

        for i in range(0, len(mlb_list), batch_size):
            batch_mlbs = mlb_list[i : i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(mlb_list) + batch_size - 1) // batch_size
            items_range = (
                f"MLBs {i+1}-{min(i+batch_size, len(mlb_list))} of {len(mlb_list)}"
            )

            log_batch_progress(batch_num, total_batches, items_range, logger)

            batch_results = process_mlb_batch_with_validation(
                batch_mlbs, original_models, feature_data, rollback_stats
            )

            updated_models.update(batch_results["models"])
            final_forecasts.update(batch_results["forecasts"])

            # Memory cleanup after each batch
            gc.collect()

        models_updated = len(updated_models)

        # Log processing results
        log_rollback_summary(rollback_stats, logger)

        # Final validation of updated models and forecasts
        if updated_models or final_forecasts:
            logger.info("Performing final validation...")
            try:
                # Create validation context for final check
                context = ValidationContext(
                    historical_stats=historical_stats, rollback_stats=rollback_stats
                )

                # Use integrated final validation
                validated_models, validated_forecasts, validation_issues = (
                    validate_final_results(updated_models, final_forecasts, context)
                )

                # Log validation results using utility function
                log_validation_summary(
                    validation_issues, len(updated_models), logger, "final validation"
                )

                # Use validated results
                updated_models = validated_models
                final_forecasts = validated_forecasts

                # Monitor memory usage during validation
                memory_stats = monitor_memory_usage_during_validation()
                log_memory_usage(memory_stats, logger, "validation")

                logger.debug(
                    f"Validation completed: {len(validated_models)} models, "
                    f"{len(validated_forecasts)} forecasts validated"
                )

            except Exception as e:
                logger.error(f"Final validation failed: {e}")
                logger.warning("Proceeding with pre-validation results...")

        # Save forecasts to database
        if final_forecasts:
            try:
                db_manager.save_forecasts_to_sql(final_forecasts)
                log_database_operation("forecasts saved", len(final_forecasts), logger)
            except Exception as e:
                log_database_operation("forecast save", 0, logger, "failed", str(e))
                # Don't fail the entire pipeline for forecast saving issues
        else:
            log_database_operation(
                "forecast save",
                0,
                logger,
                "skipped",
                "no models were updated successfully or all failed validation",
            )

        # Save updated models
        if updated_models:
            try:
                save_models(updated_models, AppConfig.MLB_REGRESSORS_FILE)
                log_model_operations("saved", len(updated_models), logger)
            except Exception as e:
                logger.error(f"Failed to save updated models: {e}")
                raise
        else:
            logger.info(
                "No updated models to save (all models were rolled back or failed)"
            )

        # Pipeline completion
        end_time = time.time()
        execution_time = end_time - start_time

        pipeline_metrics = {
            "records_processed": records_processed,
            "models_updated": models_updated,
        }

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

        # Log completion and save metadata
        pipeline_metrics["error_message"] = error_message
        log_pipeline_completion(
            final_status, execution_time, pipeline_metrics, logger, "daily"
        )

        log_pipeline_run(
            run_type=run_type,
            status=final_status,
            records_processed=records_processed,
            models_updated=models_updated,
            error_message=error_message,
            run_duration_seconds=execution_time,
        )

    except Exception as e:
        error_message = str(e)
        logger.error(f"Daily mode pipeline failed with error: {error_message}")

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
    run_type = "full"
    records_processed = 0
    models_updated = 0
    error_message = None
    use_incremental = False

    try:
        log_pipeline_start("full", logger)

        # Initialize infrastructure
        logger.debug("Initializing metadata tracking...")
        if not create_metadata_table():
            logger.warning(
                "Failed to create metadata table, continuing without metadata tracking"
            )

        if not db_manager.test_connection():
            error_message = "Database connection failed"
            logger.error(error_message)
            return

        # Check for existing models to determine training approach
        existing_models = {}
        if AppConfig.MLB_REGRESSORS_FILE.exists():
            try:
                existing_models = load_models(AppConfig.MLB_REGRESSORS_FILE)
                log_model_operations(
                    "loaded for fallback", len(existing_models), logger, level="debug"
                )
                use_incremental = True
            except Exception as e:
                logger.warning(f"Failed to load existing models: {e}")
                logger.info("Proceeding with full training instead")
                use_incremental = False
        else:
            logger.info("No existing models found - performing full training")
            use_incremental = False

        # Import complete dataset
        data = db_manager.import_data_from_sql()
        records_processed = len(data)
        log_data_info(records_processed, logger)

        # Process and prepare data
        logger.debug("Processing sales data and creating features...")
        clean_data = process_sales_data(data)
        feature_data = create_time_series_features(clean_data)
        validate_data_freshness(feature_data)

        # Log processed data info
        date_info = get_date_range_info(feature_data)
        log_data_info(date_info["record_count"], logger, date_info, "processed")

        # Decide between incremental update or full training
        if use_incremental and existing_models:
            logger.info(
                f"Training Mode: Incremental update of {len(existing_models)} models"
            )

            try:
                # Backup existing models before updating
                archive_dir = AppConfig.MLB_REGRESSORS_FILE.parent / "archive"
                archive_models(AppConfig.MLB_REGRESSORS_FILE, archive_dir)

                # Perform incremental model updates
                updated_models = update_mlb_models_incremental(
                    existing_models,
                    feature_data,
                    additional_rounds=50,  # Conservative number for incremental updates
                )
                models_updated = len(updated_models)
                log_model_operations("incrementally updated", models_updated, logger)

                # Generate forecasts using targeted approach for updated models only
                mlb_forecast, mlb_models = generate_forecasts_for_existing_models(
                    updated_models, feature_data, AppConfig.FORECAST_DAYS_LONG
                )
                logger.debug(
                    f"Generated {AppConfig.FORECAST_DAYS_LONG}-day forecasts for {len(mlb_models)} models"
                )

                logger.debug("Incremental training completed successfully")

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

        # Integrated final validation before saving
        logger.info("=== INTEGRATED FINAL VALIDATION ===")
        logger.info(
            f"Performing final validation on {len(mlb_models)} trained models and forecasts..."
        )
        try:
            # Calculate historical stats for validation
            historical_stats = calculate_historical_stats(feature_data)

            # Create validation context for final check
            rollback_stats = {
                "models_improved": 0,
                "models_maintained": 0,
                "models_rolled_back": 0,
                "mlbs_failed": 0,
                "total_processed": len(mlb_models),
            }

            context = ValidationContext(
                historical_stats=historical_stats,
                rollback_stats=rollback_stats,
                improvement_threshold=-5.0,  # Standard threshold for full training
                additional_rounds=0,  # Not applicable for full training
            )

            # Use integrated final validation
            validated_models, validated_forecasts, validation_issues = (
                validate_final_results(mlb_models, mlb_forecast, context)
            )

            # Log validation results
            if validation_issues:
                logger.warning(
                    f"Final validation found {len(validation_issues)} issues:"
                )
                for issue in validation_issues[:10]:  # Log first 10 issues
                    logger.warning(f"  - {issue}")
                if len(validation_issues) > 10:
                    logger.warning(
                        f"  ... and {len(validation_issues) - 10} more issues"
                    )
            else:
                logger.info("All models and forecasts passed final validation")

            # Use validated results
            mlb_forecast = validated_forecasts
            mlb_models = validated_models
            models_updated = len(mlb_models)

            # Monitor memory usage during validation
            memory_stats = monitor_memory_usage_during_validation()
            logger.info(
                f"Memory usage during validation: {memory_stats['rss_mb']:.1f} MB RSS, "
                f"{memory_stats['percent']:.1f}% of system memory"
            )

            logger.info(
                f"Integrated validation completed: {len(validated_models)} models, "
                f"{len(validated_forecasts)} forecasts validated"
            )

        except Exception as e:
            logger.error(f"Integrated final validation failed: {e}")
            logger.warning("Proceeding with pre-validation results...")

        # Save forecasts to database
        db_manager.save_forecasts_to_sql(mlb_forecast)
        log_database_operation("forecasts saved", len(mlb_forecast), logger)

        # Save trained/updated models for continuous learning
        save_models(mlb_models, AppConfig.MLB_REGRESSORS_FILE)
        log_model_operations("saved", len(mlb_models), logger)

        # Pipeline completion
        end_time = time.time()
        execution_time = end_time - start_time

        pipeline_metrics = {
            "records_processed": records_processed,
            "models_updated": models_updated,
        }

        # Log completion and save metadata
        log_pipeline_completion(
            "success", execution_time, pipeline_metrics, logger, "full"
        )

        log_pipeline_run(
            run_type=run_type,
            status="success",
            records_processed=records_processed,
            models_updated=models_updated,
            run_duration_seconds=execution_time,
        )

    except Exception as e:
        error_message = str(e)
        logger.error(f"Full mode pipeline failed with error: {error_message}")

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
    run_type = "incremental_filtered"
    records_processed = 0
    models_updated = 0
    error_message = None

    try:
        log_pipeline_start("since-date", logger, extra_info=f"since {since_date}")

        # Initialize infrastructure
        logger.debug("Initializing metadata tracking...")
        if not create_metadata_table():
            logger.warning(
                "Failed to create metadata table, continuing without metadata tracking"
            )

        if not db_manager.test_connection():
            error_message = "Database connection failed"
            logger.error(error_message)
            return

        # Load existing models for incremental updates
        if not AppConfig.MLB_REGRESSORS_FILE.exists():
            error_message = f"No existing models found at {AppConfig.MLB_REGRESSORS_FILE}. Run full training first."
            logger.error(error_message)
            raise FileNotFoundError(error_message)

        existing_models = load_models(AppConfig.MLB_REGRESSORS_FILE)
        log_model_operations("loaded", len(existing_models), logger)

        # Import incremental data since specified date
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
                f"No new data since {since_date} - pipeline completed with no updates"
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
        log_data_info(
            records_processed, logger, operation=f"imported since {since_date}"
        )

        # Merge with historical data and process
        historical_data_path = AppConfig.RAW_DATA_FILE
        if historical_data_path.exists():
            merged_data = merge_with_historical(new_data, historical_data_path)
            logger.info(
                f"Merged with historical data: {len(merged_data):,} total records"
            )
        else:
            logger.warning("No historical data file found, using only new data")
            merged_data = new_data

        # Process and prepare data
        logger.debug("Processing sales data and creating features...")
        clean_data = process_sales_data(merged_data)
        feature_data = create_time_series_features(clean_data)

        # Backup and perform incremental model updates
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
        log_model_operations("incrementally updated", models_updated, logger)

        # Generate forecasts for updated models
        mlb_forecast, mlb_models = generate_forecasts_for_existing_models(
            updated_models, feature_data, AppConfig.FORECAST_DAYS_LONG
        )
        logger.debug(
            f"Generated {AppConfig.FORECAST_DAYS_LONG}-day forecasts for {len(mlb_forecast)} MLBs"
        )

        # Integrated final validation before saving
        logger.info("=== INTEGRATED FINAL VALIDATION ===")
        logger.info(
            f"Performing final validation on {len(mlb_models)} updated models and forecasts..."
        )
        try:
            # Calculate historical stats for validation
            historical_stats = calculate_historical_stats(feature_data)

            # Create validation context for final check
            rollback_stats = {
                "models_improved": 0,
                "models_maintained": 0,
                "models_rolled_back": 0,
                "mlbs_failed": 0,
                "total_processed": len(mlb_models),
            }

            context = ValidationContext(
                historical_stats=historical_stats,
                rollback_stats=rollback_stats,
                improvement_threshold=-5.0,  # Standard threshold for since-date mode
                additional_rounds=0,  # Not applicable for forecast validation
            )

            # Use integrated final validation
            validated_models, validated_forecasts, validation_issues = (
                validate_final_results(mlb_models, mlb_forecast, context)
            )

            # Log validation results
            if validation_issues:
                logger.warning(
                    f"Final validation found {len(validation_issues)} issues:"
                )
                for issue in validation_issues[:10]:  # Log first 10 issues
                    logger.warning(f"  - {issue}")
                if len(validation_issues) > 10:
                    logger.warning(
                        f"  ... and {len(validation_issues) - 10} more issues"
                    )
            else:
                logger.info("All models and forecasts passed final validation")

            # Use validated results
            mlb_forecast = validated_forecasts
            mlb_models = validated_models
            models_updated = len(mlb_models)

            # Monitor memory usage during validation
            memory_stats = monitor_memory_usage_during_validation()
            logger.info(
                f"Memory usage during validation: {memory_stats['rss_mb']:.1f} MB RSS, "
                f"{memory_stats['percent']:.1f}% of system memory"
            )

            logger.info(
                f"Integrated validation completed: {len(validated_models)} models, "
                f"{len(validated_forecasts)} forecasts validated"
            )

        except Exception as e:
            logger.error(f"Integrated final validation failed: {e}")
            logger.warning("Proceeding with pre-validation results...")

        # Save forecasts to database
        if mlb_forecast:
            logger.info("Saving forecasts to remote SQL database...")
            db_manager.save_forecasts_to_sql(mlb_forecast)
            logger.info("Forecasts saved successfully!")

        # Save updated models
        logger.info("Saving updated models for continuous learning...")
        save_models(mlb_models, AppConfig.MLB_REGRESSORS_FILE)
        logger.info(
            f"Saved {len(mlb_models)} updated models to {AppConfig.MLB_REGRESSORS_FILE}"
        )

        # Update historical data file with new data
        save_merged_data(merged_data, historical_data_path)
        logger.debug(
            f"Updated historical data file with {records_processed:,} new records"
        )

        # Pipeline completion
        end_time = time.time()
        execution_time = end_time - start_time

        pipeline_metrics = {
            "records_processed": records_processed,
            "models_updated": models_updated,
        }

        # Log completion and save metadata
        log_pipeline_completion(
            "success", execution_time, pipeline_metrics, logger, "since-date"
        )

        log_pipeline_run(
            run_type=run_type,
            status="success",
            records_processed=records_processed,
            models_updated=models_updated,
            run_duration_seconds=execution_time,
        )

    except Exception as e:
        error_message = str(e)
        logger.error(f"Since-date mode pipeline failed: {error_message}")

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


def run_monthly_mode():
    """Execute monthly mode: full retrain with sliding window (last 6 months)."""
    start_time = time.time()
    run_type = "monthly"
    records_processed = 0
    models_updated = 0
    error_message = None

    try:
        log_pipeline_start("monthly", logger)

        # Initialize infrastructure
        logger.debug("Initializing metadata tracking...")
        if not create_metadata_table():
            logger.warning(
                "Failed to create metadata table, continuing without metadata tracking"
            )

        if not db_manager.test_connection():
            error_message = "Database connection failed"
            logger.error(error_message)
            return

        # Load existing models for comparison
        existing_models = {}
        if AppConfig.MLB_REGRESSORS_FILE.exists():
            try:
                existing_models = load_models(AppConfig.MLB_REGRESSORS_FILE)
                log_model_operations(
                    "loaded for comparison", len(existing_models), logger, level="debug"
                )
            except Exception as e:
                logger.warning(f"Failed to load existing models: {e}")
                logger.info("Proceeding with monthly training without model comparison")
        else:
            logger.info("No existing models found - training all models from scratch")

        # Import 6-month sliding window data
        data = import_data_last_n_months(
            user=DatabaseConfig.USER,
            password=DatabaseConfig.PASSWORD,
            host=DatabaseConfig.HOST,
            port=DatabaseConfig.PORT,
            dbname=DatabaseConfig.DBNAME,
            view=DatabaseConfig.VIEW,
            months=6,
        )
        records_processed = len(data)
        log_data_info(
            records_processed, logger, operation="imported (6-month sliding window)"
        )

        # Process and prepare data
        logger.debug("Processing sales data and creating features...")
        clean_data = process_sales_data(data)
        feature_data = create_time_series_features(clean_data)
        validate_data_freshness(feature_data)

        # Log processed data info
        date_info = get_date_range_info(feature_data)
        log_data_info(date_info["record_count"], logger, date_info, "processed")

        # Train models with 6-month sliding window
        logger.info("Training Mode: Monthly retraining (6-month sliding window)")

        # Generate forecasts and train models from scratch with lookback filter
        mlb_forecast, mlb_models = forecast_future_sales_direct(
            feature_data, AppConfig.FORECAST_DAYS_LONG, lookback_months=6
        )
        models_updated = len(mlb_models)
        log_model_operations("trained with sliding window", models_updated, logger)

        # Model comparison and validation if existing models are available
        if existing_models:
            logger.info(
                f"Comparing {len(mlb_models)} new models against {len(existing_models)} existing models"
            )

            # Calculate historical stats for validation
            historical_stats = calculate_historical_stats(feature_data)

            # Create validation context for model comparison
            rollback_stats = {
                "models_improved": 0,
                "models_maintained": 0,
                "models_rolled_back": 0,
                "mlbs_failed": 0,
                "total_processed": len(mlb_models),
            }

            context = ValidationContext(
                historical_stats=historical_stats,
                rollback_stats=rollback_stats,
                improvement_threshold=-2.0,  # Allow up to 2% degradation for monthly retraining
                additional_rounds=0,  # Not applicable for full retraining
            )

            # Use integrated monthly retraining validation with systematic model comparison
            final_models, validation_results, validation_issues = (
                validate_monthly_retraining_results(
                    old_models=existing_models,
                    new_models=mlb_models,
                    feature_data=feature_data,
                    context=context,
                )
            )

            # Generate forecasts for final selected models using existing functionality
            final_forecasts, _ = generate_forecasts_for_existing_models(
                final_models, feature_data, AppConfig.FORECAST_DAYS_LONG
            )
            logger.debug(f"Generated forecasts for {len(final_models)} final models")

            # Log validation results using utility function
            log_validation_summary(
                validation_issues, len(final_models), logger, "model comparison"
            )

            # Update variables for consistency with rest of function
            mlb_models = final_models
            mlb_forecast = final_forecasts
            models_updated = len(mlb_models)

            # Log comparison statistics
            log_rollback_summary(context.rollback_stats, logger, "monthly comparison")

            # Log additional comparison details if available
            log_model_comparison_results(
                validation_results, logger, "monthly retraining"
            )
        else:
            logger.info("No existing models found for comparison, using all new models")
            final_forecasts, _ = generate_forecasts_for_existing_models(
                mlb_models, feature_data, AppConfig.FORECAST_DAYS_LONG
            )
            mlb_forecast = final_forecasts

        # Save forecasts to database
        db_manager.save_forecasts_to_sql(mlb_forecast)
        log_database_operation("forecasts saved", len(mlb_forecast), logger)

        # Backup existing models before saving new ones
        if AppConfig.MLB_REGRESSORS_FILE.exists():
            try:
                archive_dir = AppConfig.MLB_REGRESSORS_FILE.parent / "archive"
                archive_models(AppConfig.MLB_REGRESSORS_FILE, archive_dir)
                logger.debug(f"Models backed up to {archive_dir}")
            except Exception as e:
                logger.warning(f"Failed to backup models: {e}. Continuing with save...")

        # Save trained/updated models
        save_models(mlb_models, AppConfig.MLB_REGRESSORS_FILE)
        log_model_operations("saved", len(mlb_models), logger)

        # Pipeline completion
        end_time = time.time()
        execution_time = end_time - start_time

        pipeline_metrics = {
            "records_processed": records_processed,
            "models_updated": models_updated,
        }

        # Log completion and save metadata
        log_pipeline_completion(
            "success", execution_time, pipeline_metrics, logger, "monthly"
        )

        log_pipeline_run(
            run_type=run_type,
            status="success",
            records_processed=records_processed,
            models_updated=models_updated,
            run_duration_seconds=execution_time,
        )

    except Exception as e:
        error_message = str(e)
        logger.error(f"Monthly mode pipeline failed with error: {error_message}")

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
    """Main function with unified CLI interface for all pipeline modes."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Unified pipeline runner for continuous learning with multiple execution modes"
    )
    parser.add_argument(
        "--mode",
        choices=["daily", "full", "monthly"],
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
        logger.info(f"Running since-date mode with date: {since_date}")
        run_since_date_mode(since_date)
    elif args.mode == "daily":
        logger.info("Running daily mode pipeline")
        run_daily_mode()
    elif args.mode == "full":
        logger.info("Running full mode pipeline")
        run_full_mode()
    elif args.mode == "monthly":
        logger.info("Running monthly mode pipeline")
        run_monthly_mode()
    else:
        logger.error(f"Unknown mode: {args.mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
