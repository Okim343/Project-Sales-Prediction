"""
Integrated validation for continuous learning pipeline.

This module provides unified validation functions that integrate model validation,
forecast validation, and rollback logic into the main pipeline flow, addressing
the disconnected validation issue identified in suggestions-for-later.md.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, NamedTuple
from dataclasses import dataclass

from estimation.model_forecast import (
    update_mlb_models_incremental,
    validate_model_improvement,
)
from validation.model_validator import (
    validate_model_consistency,
)
from validation.forecast_validator import (
    validate_forecasts,
)

logger = logging.getLogger(__name__)


@dataclass
class ValidationContext:
    """Context object to track validation state across pipeline operations."""

    historical_stats: Dict[str, Dict]
    rollback_stats: Dict[str, int]
    improvement_threshold: float = -5.0  # Allow up to 5% degradation
    additional_rounds: int = 25  # Conservative rounds for daily updates

    def __post_init__(self):
        """Initialize rollback stats if not provided."""
        if not self.rollback_stats:
            self.rollback_stats = {
                "models_improved": 0,
                "models_maintained": 0,
                "models_rolled_back": 0,
                "mlbs_failed": 0,
                "total_processed": 0,
            }


class MLBValidationResult(NamedTuple):
    """Result from validating and updating a single MLB."""

    mlb: str
    success: bool
    updated_model: Optional[object] = None
    forecast_tuple: Optional[Tuple] = None
    validation_metrics: Optional[Dict] = None
    issues: Optional[list] = None
    decision: str = "failed"  # improved, maintained, rolled_back, failed


def validate_and_update_mlb_model(
    mlb: str,
    original_model: object,
    mlb_data: pd.DataFrame,
    context: ValidationContext,
) -> MLBValidationResult:
    """
    Integrated function that performs model update, validation, and rollback decision
    for a single MLB in one unified flow.

    Args:
        mlb: MLB code to process
        original_model: Original model for this MLB
        mlb_data: Data for this specific MLB
        context: Validation context with historical stats and settings

    Returns:
        MLBValidationResult with update outcome and validation metrics
    """
    try:
        logger.info(f"Processing MLB {mlb} with integrated validation...")

        # Check minimum data requirement
        if len(mlb_data) < 10:
            logger.warning(
                f"MLB {mlb}: Insufficient data ({len(mlb_data)} rows), skipping"
            )
            context.rollback_stats["mlbs_failed"] += 1
            return MLBValidationResult(
                mlb=mlb,
                success=False,
                issues=[f"Insufficient data: {len(mlb_data)} rows"],
                decision="failed",
            )

        # Step 1: Perform incremental model update
        try:
            logger.debug(f"MLB {mlb}: Performing incremental update...")
            updated_model = update_mlb_models_incremental(
                {mlb: original_model},
                mlb_data,
                additional_rounds=context.additional_rounds,
            )[mlb]
        except Exception as update_error:
            logger.error(f"MLB {mlb}: Incremental update failed: {update_error}")
            context.rollback_stats["mlbs_failed"] += 1
            return MLBValidationResult(
                mlb=mlb,
                success=False,
                issues=[f"Model update failed: {update_error}"],
                decision="failed",
            )

        # Step 2: Create validation data and validate model improvement
        validation_metrics = None
        decision = "maintained"

        try:
            # Create validation data from the last 30% of MLB data
            val_size = max(10, len(mlb_data) // 3)
            val_data = mlb_data.iloc[-val_size:]

            # Prepare validation features and targets for 90-day forecasting
            FEATURES = ["day_of_week", "day_of_month", "rolling_mean_3", "lag_1"]
            from config import AppConfig

            forecast_days = AppConfig.FORECAST_DAYS_LONG

            # Create validation data similar to training format
            X_val, y_val = _create_validation_data(val_data, FEATURES, forecast_days)

            if len(X_val) > 0:
                # Validate model improvement
                validation_result = validate_model_improvement(
                    original_model, updated_model, (X_val, y_val)
                )
                validation_metrics = validation_result

                # Make rollback decision based on improvement threshold
                if (
                    validation_result["improvement_percentage"]
                    >= context.improvement_threshold
                ):
                    if validation_result["improvement_percentage"] > 0:
                        context.rollback_stats["models_improved"] += 1
                        decision = "improved"
                    else:
                        context.rollback_stats["models_maintained"] += 1
                        decision = "maintained"

                    logger.info(
                        f"MLB {mlb}: Model {decision} "
                        f"({validation_result['improvement_percentage']:.2f}% change), keeping update"
                    )

                    # Step 3: Generate and validate forecast for accepted model
                    forecast_result = validate_and_generate_forecast(
                        mlb, updated_model, mlb_data, context
                    )

                    return MLBValidationResult(
                        mlb=mlb,
                        success=True,
                        updated_model=updated_model,
                        forecast_tuple=forecast_result,
                        validation_metrics=validation_metrics,
                        decision=decision,
                    )
                else:
                    # Rollback decision
                    context.rollback_stats["models_rolled_back"] += 1
                    logger.info(
                        f"MLB {mlb}: Model degraded "
                        f"({validation_result['improvement_percentage']:.2f}% change), rolling back"
                    )

                    return MLBValidationResult(
                        mlb=mlb,
                        success=False,
                        validation_metrics=validation_metrics,
                        issues=[
                            f"Performance degraded by {validation_result['improvement_percentage']:.2f}%"
                        ],
                        decision="rolled_back",
                    )
            else:
                # Not enough validation data - keep update with warning
                context.rollback_stats["models_maintained"] += 1
                logger.warning(
                    f"MLB {mlb}: Insufficient validation data, keeping update by default"
                )

                # Generate forecast for maintained model
                forecast_result = validate_and_generate_forecast(
                    mlb, updated_model, mlb_data, context
                )

                return MLBValidationResult(
                    mlb=mlb,
                    success=True,
                    updated_model=updated_model,
                    forecast_tuple=forecast_result,
                    validation_metrics={"note": "insufficient_validation_data"},
                    issues=["Insufficient validation data"],
                    decision="maintained",
                )

        except Exception as validation_error:
            logger.error(f"MLB {mlb}: Validation failed: {validation_error}")
            context.rollback_stats["mlbs_failed"] += 1
            return MLBValidationResult(
                mlb=mlb,
                success=False,
                issues=[f"Validation failed: {validation_error}"],
                decision="failed",
            )

    except Exception as e:
        logger.error(f"MLB {mlb}: Integrated validation failed: {e}")
        context.rollback_stats["mlbs_failed"] += 1
        return MLBValidationResult(
            mlb=mlb,
            success=False,
            issues=[f"Processing failed: {e}"],
            decision="failed",
        )


def validate_and_generate_forecast(
    mlb: str,
    model: object,
    mlb_data: pd.DataFrame,
    context: ValidationContext,
) -> Optional[Tuple]:
    """
    Generate and validate forecast for a single MLB in one integrated step.

    Args:
        mlb: MLB code
        model: Trained/updated model
        mlb_data: Data for this MLB
        context: Validation context with historical stats

    Returns:
        Validated forecast tuple (forecast_df, sku) or None if validation failed
    """
    try:
        # Step 1: Generate forecast
        forecast_tuple = _generate_forecast_for_mlb(mlb, model, mlb_data)
        if not forecast_tuple:
            return None

        forecast_df, sku = forecast_tuple

        # Step 2: Validate individual forecast
        validated_forecast = _validate_single_forecast(
            mlb, forecast_df, sku, context.historical_stats
        )

        if validated_forecast:
            logger.debug(f"MLB {mlb}: Forecast generated and validated successfully")
            return validated_forecast
        else:
            logger.warning(f"MLB {mlb}: Forecast failed validation")
            return None

    except Exception as e:
        logger.error(f"MLB {mlb}: Forecast generation/validation failed: {e}")
        return None


def process_mlb_batch_integrated(
    batch_mlbs: list,
    original_models: Dict,
    feature_data: pd.DataFrame,
    context: ValidationContext,
) -> Dict:
    """
    Process a batch of MLBs using integrated validation approach.

    This replaces the previous process_mlb_batch_with_validation function
    with a more integrated approach that handles validation inline.

    Args:
        batch_mlbs: List of MLB codes to process
        original_models: Dictionary of original models
        feature_data: Full dataset with features
        context: Validation context

    Returns:
        Dictionary with 'models' and 'forecasts' keys containing successful updates
    """
    batch_results = {"models": {}, "forecasts": {}}

    for mlb in batch_mlbs:
        try:
            # Get data for this specific MLB
            mlb_data = feature_data[feature_data["mlb"] == mlb].copy()
            if mlb not in original_models:
                logger.warning(f"MLB {mlb}: No original model found, skipping")
                context.rollback_stats["mlbs_failed"] += 1
                continue

            # Get original model
            original_model = original_models[mlb]

            # Process with integrated validation
            result = validate_and_update_mlb_model(
                mlb, original_model, mlb_data, context
            )

            # Handle results
            if result.success and result.updated_model and result.forecast_tuple:
                batch_results["models"][mlb] = result.updated_model
                batch_results["forecasts"][mlb] = result.forecast_tuple
                logger.debug(
                    f"MLB {mlb}: Successfully processed with integrated validation"
                )
            else:
                # Log the reason for failure/rollback
                if result.issues:
                    for issue in result.issues:
                        logger.warning(f"MLB {mlb}: {issue}")

        except Exception as e:
            logger.error(f"MLB {mlb}: Batch processing failed: {e}")
            context.rollback_stats["mlbs_failed"] += 1
            continue

    return batch_results


def validate_final_results(
    updated_models: Dict,
    final_forecasts: Dict,
    context: ValidationContext,
) -> Tuple[Dict, Dict, list]:
    """
    Perform final validation on batch results with integrated consistency checks.

    Args:
        updated_models: Dictionary of updated models
        final_forecasts: Dictionary of generated forecasts
        context: Validation context

    Returns:
        Tuple of (validated_models, validated_forecasts, issues_list)
    """
    logger.info("Performing final integrated validation on batch results...")
    issues_list = []

    # Step 1: Model consistency validation
    if updated_models:
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
                    issues_list.append(f"Model consistency: {issue}")
            else:
                logger.info("All models passed consistency validation")
        except Exception as e:
            error_msg = f"Model consistency validation failed: {e}"
            logger.error(error_msg)
            issues_list.append(error_msg)

    # Step 2: Batch forecast validation (for any remaining forecasts)
    validated_forecasts = final_forecasts
    if final_forecasts:
        try:
            # Use batch validation as a final check
            batch_validated, validation_issues = validate_forecasts(
                final_forecasts, context.historical_stats
            )

            if validation_issues:
                logger.info(
                    f"Final forecast validation found {len(validation_issues)} additional issues:"
                )
                for issue in validation_issues[:5]:  # Log first 5 issues
                    logger.info(f"  - {issue}")
                    issues_list.extend(validation_issues)

            validated_forecasts = batch_validated
            logger.info(
                f"Final forecast validation: {len(validated_forecasts)}/{len(final_forecasts)} forecasts validated"
            )

        except Exception as e:
            error_msg = f"Final forecast validation failed: {e}"
            logger.error(error_msg)
            issues_list.append(error_msg)

    return updated_models, validated_forecasts, issues_list


# Helper functions


def _create_validation_data(
    val_data: pd.DataFrame, features: list, forecast_days: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create validation features and targets from MLB data."""
    X_val = []
    y_val = []

    for i in range(len(val_data) - forecast_days):
        X_val.append(val_data.iloc[i][features].values)
        y_val.append(val_data.iloc[i + 1 : i + forecast_days + 1]["quant"].values)

    if len(X_val) > 0:
        X_val = pd.DataFrame(X_val, columns=features)
        y_columns = [f"quant_{i+1}" for i in range(forecast_days)]
        y_val = pd.DataFrame(y_val, columns=y_columns)
    else:
        X_val = pd.DataFrame(columns=features)
        y_val = pd.DataFrame()

    return X_val, y_val


def _generate_forecast_for_mlb(
    mlb: str, model: object, mlb_data: pd.DataFrame
) -> Optional[Tuple]:
    """Generate forecast for a single MLB using its model."""
    try:
        from config import AppConfig

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

        # Round predictions to nearest integers
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


def _validate_single_forecast(
    mlb: str, forecast_df: pd.DataFrame, sku: str, historical_stats: Dict
) -> Optional[Tuple]:
    """Validate a single forecast inline and return corrected version or None."""
    try:
        # Use the existing batch validator for a single forecast
        single_forecast_dict = {mlb: (forecast_df, sku)}
        validated_dict, issues = validate_forecasts(
            single_forecast_dict, historical_stats
        )

        if mlb in validated_dict:
            if issues:
                logger.debug(
                    f"MLB {mlb}: Forecast had {len(issues)} issues but was corrected"
                )
            return validated_dict[mlb]
        else:
            logger.warning(f"MLB {mlb}: Forecast failed validation and was rejected")
            return None

    except Exception as e:
        logger.error(f"MLB {mlb}: Forecast validation failed: {e}")
        return None
