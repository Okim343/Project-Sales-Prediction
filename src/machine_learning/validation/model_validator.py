"""Enhanced model validation for continuous learning pipeline."""

import logging
import psutil
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error


def validate_daily_updates(
    original_models: Dict, updated_models: Dict, validation_data: Dict
) -> Dict[str, Dict]:
    """
    Validate incremental model updates and provide rollback recommendations.

    Args:
        original_models: Dictionary of original models by MLB
        updated_models: Dictionary of updated models by MLB
        validation_data: Dictionary of validation data by MLB
                        Format: {mlb: (X_val, y_val)}

    Returns:
        Dictionary with validation results and rollback recommendations
        Format: {
            'rollback_recommendations': {mlb: bool},
            'performance_metrics': {mlb: {...}},
            'memory_usage': {...},
            'summary': {...}
        }
    """
    logger = logging.getLogger(__name__)
    results = {
        "rollback_recommendations": {},
        "performance_metrics": {},
        "memory_usage": {},
        "summary": {
            "total_models": 0,
            "improved_models": 0,
            "degraded_models": 0,
            "rollback_count": 0,
            "validation_errors": 0,
        },
    }

    logger.info(f"Validating {len(updated_models)} model updates...")

    # Monitor memory usage during validation
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB

    results["summary"]["total_models"] = len(updated_models)

    for mlb in updated_models.keys():
        if mlb not in original_models:
            logger.warning(f"MLB {mlb}: No original model found, skipping validation")
            continue

        if mlb not in validation_data:
            logger.warning(f"MLB {mlb}: No validation data found, skipping validation")
            continue

        try:
            original_model = original_models[mlb]
            updated_model = updated_models[mlb]
            X_val, y_val = validation_data[mlb]

            # Perform model comparison
            metrics = compare_model_performance(
                original_model, updated_model, X_val, y_val, mlb
            )

            results["performance_metrics"][mlb] = metrics

            # Determine rollback recommendation
            improvement_threshold = -5.0  # Allow up to 5% degradation

            if metrics["improvement_percentage"] >= improvement_threshold:
                results["rollback_recommendations"][mlb] = False  # Keep updated model
                if metrics["improvement_percentage"] > 0:
                    results["summary"]["improved_models"] += 1
            else:
                results["rollback_recommendations"][mlb] = True  # Rollback to original
                results["summary"]["degraded_models"] += 1
                results["summary"]["rollback_count"] += 1
                logger.warning(
                    f"MLB {mlb}: Recommending rollback due to {metrics['improvement_percentage']:.2f}% degradation"
                )

        except Exception as e:
            logger.error(f"MLB {mlb}: Validation failed: {e}")
            results["rollback_recommendations"][mlb] = True  # Rollback on error
            results["summary"]["validation_errors"] += 1
            results["summary"]["rollback_count"] += 1

    # Record final memory usage
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    results["memory_usage"] = {
        "initial_mb": round(initial_memory, 1),
        "final_mb": round(final_memory, 1),
        "peak_increase_mb": round(final_memory - initial_memory, 1),
    }

    # Log summary
    summary = results["summary"]
    logger.info("=== MODEL VALIDATION SUMMARY ===")
    logger.info(f"Total models validated: {summary['total_models']}")
    logger.info(f"Improved models: {summary['improved_models']}")
    logger.info(f"Degraded models: {summary['degraded_models']}")
    logger.info(f"Rollback recommendations: {summary['rollback_count']}")
    logger.info(f"Validation errors: {summary['validation_errors']}")
    logger.info(
        f"Memory usage: {results['memory_usage']['peak_increase_mb']} MB increase"
    )

    return results


def compare_model_performance(
    original_model, updated_model, X_val: pd.DataFrame, y_val: pd.DataFrame, mlb: str
) -> Dict:
    """
    Compare performance between original and updated models.

    Args:
        original_model: Original trained model
        updated_model: Updated trained model
        X_val: Validation features
        y_val: Validation targets
        mlb: MLB identifier for logging

    Returns:
        Dictionary with performance comparison metrics
    """
    logger = logging.getLogger(__name__)

    try:
        # Generate predictions from both models
        orig_pred = original_model.predict(X_val)
        updated_pred = updated_model.predict(X_val)

        # Handle multi-output predictions if necessary
        if len(orig_pred.shape) > 1 and orig_pred.shape[1] > 1:
            # Multi-output case - flatten predictions for comparison
            orig_pred = orig_pred.flatten()
            updated_pred = updated_pred.flatten()
            y_val_flat = y_val.values.flatten()
        else:
            # Single output case
            if len(orig_pred.shape) > 1:
                orig_pred = orig_pred.flatten()
            if len(updated_pred.shape) > 1:
                updated_pred = updated_pred.flatten()
            y_val_flat = y_val.values.flatten()

        # Calculate metrics for original model
        orig_rmse = np.sqrt(mean_squared_error(y_val_flat, orig_pred))
        orig_mae = mean_absolute_error(y_val_flat, orig_pred)

        # Calculate metrics for updated model
        updated_rmse = np.sqrt(mean_squared_error(y_val_flat, updated_pred))
        updated_mae = mean_absolute_error(y_val_flat, updated_pred)

        # Calculate improvement percentages
        rmse_improvement = (
            ((orig_rmse - updated_rmse) / orig_rmse) * 100 if orig_rmse > 0 else 0
        )
        mae_improvement = (
            ((orig_mae - updated_mae) / orig_mae) * 100 if orig_mae > 0 else 0
        )

        # Use RMSE as primary metric for improvement decision
        primary_improvement = rmse_improvement

        metrics = {
            "original_rmse": float(orig_rmse),
            "updated_rmse": float(updated_rmse),
            "original_mae": float(orig_mae),
            "updated_mae": float(updated_mae),
            "rmse_improvement_percentage": float(rmse_improvement),
            "mae_improvement_percentage": float(mae_improvement),
            "improvement_percentage": float(primary_improvement),
            "validation_samples": len(y_val_flat),
        }

        logger.debug(
            f"MLB {mlb}: RMSE {orig_rmse:.2f} -> {updated_rmse:.2f} "
            f"({rmse_improvement:+.2f}%), MAE {orig_mae:.2f} -> {updated_mae:.2f} "
            f"({mae_improvement:+.2f}%)"
        )

        return metrics

    except Exception as e:
        logger.error(f"MLB {mlb}: Error comparing models: {e}")
        # Return neutral metrics on error to trigger rollback
        return {
            "original_rmse": float("inf"),
            "updated_rmse": float("inf"),
            "original_mae": float("inf"),
            "updated_mae": float("inf"),
            "rmse_improvement_percentage": -100.0,
            "mae_improvement_percentage": -100.0,
            "improvement_percentage": -100.0,
            "validation_samples": 0,
            "error": str(e),
        }


def validate_model_consistency(models: Dict) -> Tuple[bool, List[str]]:
    """
    Validate model consistency and structure.

    Args:
        models: Dictionary of models to validate

    Returns:
        Tuple of (is_valid, issues_list)
    """
    logger = logging.getLogger(__name__)
    issues = []

    try:
        if not models:
            issues.append("No models provided for validation")
            return False, issues

        # Check for model type consistency
        model_types = set()
        for mlb, model in models.items():
            model_type = type(model).__name__
            model_types.add(model_type)

            # Basic model health checks
            if not hasattr(model, "predict"):
                issues.append(f"MLB {mlb}: Model missing predict method")

            # XGBoost specific checks
            if hasattr(model, "get_booster"):
                try:
                    booster = model.get_booster()
                    if booster is None:
                        issues.append(f"MLB {mlb}: XGBoost model has no booster")
                except Exception as e:
                    issues.append(f"MLB {mlb}: Error accessing XGBoost booster: {e}")

        # Warn if mixed model types (unusual but not necessarily invalid)
        if len(model_types) > 1:
            issues.append(f"Mixed model types detected: {model_types}")

        logger.info(f"Model consistency check: {len(issues)} issues found")
        return len(issues) == 0, issues

    except Exception as e:
        logger.error(f"Model consistency validation failed: {e}")
        return False, [f"Validation error: {e}"]


def monitor_memory_usage_during_validation() -> Dict[str, float]:
    """
    Monitor memory usage during model validation.

    Returns:
        Dictionary with memory usage statistics
    """
    try:
        process = psutil.Process()
        memory_info = process.memory_info()

        return {
            "rss_mb": memory_info.rss / 1024 / 1024,  # Resident Set Size
            "vms_mb": memory_info.vms / 1024 / 1024,  # Virtual Memory Size
            "percent": process.memory_percent(),
            "available_mb": psutil.virtual_memory().available / 1024 / 1024,
        }
    except Exception:
        return {"rss_mb": 0.0, "vms_mb": 0.0, "percent": 0.0, "available_mb": 0.0}


def validate_prediction_quality(
    predictions: np.ndarray,
    actuals: np.ndarray,
    mlb: str,
    tolerance_factor: float = 2.0,
) -> Dict[str, any]:
    """
    Validate prediction quality against actuals.

    Args:
        predictions: Model predictions
        actuals: Actual values
        mlb: MLB identifier
        tolerance_factor: Factor for acceptable prediction range

    Returns:
        Dictionary with prediction quality metrics
    """
    logger = logging.getLogger(__name__)

    try:
        # Basic quality checks
        quality_metrics = {
            "mean_prediction": float(np.mean(predictions)),
            "mean_actual": float(np.mean(actuals)),
            "prediction_std": float(np.std(predictions)),
            "actual_std": float(np.std(actuals)),
            "correlation": float(np.corrcoef(predictions, actuals)[0, 1])
            if len(predictions) > 1
            else 0.0,
            "within_tolerance_percent": 0.0,
            "quality_score": 0.0,
        }

        # Check percentage of predictions within tolerance
        if len(actuals) > 0:
            actual_mean = np.mean(actuals)
            tolerance_range = actual_mean * tolerance_factor

            within_tolerance = np.abs(predictions - actuals) <= tolerance_range
            quality_metrics["within_tolerance_percent"] = float(
                np.mean(within_tolerance) * 100
            )

        # Calculate overall quality score (0-100)
        correlation_score = max(0, quality_metrics["correlation"]) * 50  # Max 50 points
        tolerance_score = (
            quality_metrics["within_tolerance_percent"] * 0.5
        )  # Max 50 points
        quality_metrics["quality_score"] = correlation_score + tolerance_score

        logger.debug(
            f"MLB {mlb}: Prediction quality score: {quality_metrics['quality_score']:.1f}/100"
        )

        return quality_metrics

    except Exception as e:
        logger.error(f"MLB {mlb}: Error validating prediction quality: {e}")
        return {
            "mean_prediction": 0.0,
            "mean_actual": 0.0,
            "prediction_std": 0.0,
            "actual_std": 0.0,
            "correlation": 0.0,
            "within_tolerance_percent": 0.0,
            "quality_score": 0.0,
            "error": str(e),
        }


def create_validation_data(
    mlb_data: pd.DataFrame, features: List[str], forecast_days: int
) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Create validation data for a single MLB for integrated validation.

    Args:
        mlb_data: Data for a specific MLB
        features: List of feature column names
        forecast_days: Number of days to forecast

    Returns:
        Tuple of (X_val, y_val) or None if insufficient data
    """
    logger = logging.getLogger(__name__)

    try:
        # Use last 30% of data for validation, minimum 10 samples
        val_size = max(10, len(mlb_data) // 3)

        if val_size >= len(mlb_data) or val_size < 5:
            logger.warning(
                f"Insufficient data for validation: {len(mlb_data)} total, {val_size} validation"
            )
            return None

        val_data = mlb_data.iloc[-val_size:]

        # Create validation features and targets
        X_val = []
        y_val = []

        for i in range(len(val_data) - forecast_days):
            if i >= 0 and i + forecast_days < len(val_data):
                X_val.append(val_data.iloc[i][features].values)
                y_val.append(
                    val_data.iloc[i + 1 : i + forecast_days + 1]["quant"].values
                )

        if len(X_val) == 0:
            logger.warning("No valid validation samples could be created")
            return None

        # Convert to DataFrames
        X_val = pd.DataFrame(X_val, columns=features)
        y_columns = [f"quant_{i+1}" for i in range(forecast_days)]
        y_val = pd.DataFrame(y_val, columns=y_columns)

        logger.debug(
            f"Created validation data: {len(X_val)} samples with {len(features)} features"
        )
        return X_val, y_val

    except Exception as e:
        logger.error(f"Failed to create validation data: {e}")
        return None


def validate_single_model_consistency(
    model: object, mlb: str
) -> Tuple[bool, List[str]]:
    """
    Validate consistency and structure of a single model for integrated validation.

    Args:
        model: Model to validate
        mlb: MLB identifier for logging

    Returns:
        Tuple of (is_valid, issues_list)
    """
    logger = logging.getLogger(__name__)
    issues = []

    try:
        # Basic model health checks
        if not hasattr(model, "predict"):
            issues.append(f"MLB {mlb}: Model missing predict method")

        # Check if model is fitted (has some state)
        if hasattr(model, "n_features_in_"):
            if model.n_features_in_ <= 0:
                issues.append(
                    f"MLB {mlb}: Model appears to be unfitted (n_features_in_ <= 0)"
                )

        # XGBoost specific checks
        if hasattr(model, "get_booster"):
            try:
                booster = model.get_booster()
                if booster is None:
                    issues.append(f"MLB {mlb}: XGBoost model has no booster")
            except Exception as e:
                issues.append(f"MLB {mlb}: Error accessing XGBoost booster: {e}")

        # MultiOutputRegressor specific checks
        if hasattr(model, "estimators_"):
            try:
                estimators = model.estimators_
                if not estimators or len(estimators) == 0:
                    issues.append(f"MLB {mlb}: MultiOutputRegressor has no estimators")
                else:
                    # Check each estimator
                    for i, estimator in enumerate(estimators):
                        if not hasattr(estimator, "predict"):
                            issues.append(
                                f"MLB {mlb}: Estimator {i} missing predict method"
                            )
            except Exception as e:
                issues.append(
                    f"MLB {mlb}: Error checking MultiOutputRegressor estimators: {e}"
                )

        logger.debug(f"MLB {mlb}: Model consistency check found {len(issues)} issues")
        return len(issues) == 0, issues

    except Exception as e:
        logger.error(f"MLB {mlb}: Model consistency validation failed: {e}")
        return False, [f"Validation error: {e}"]


def compare_single_model_performance(
    original_model: object,
    updated_model: object,
    X_val: pd.DataFrame,
    y_val: pd.DataFrame,
    mlb: str,
    improvement_threshold: float = -5.0,
) -> Dict[str, any]:
    """
    Compare performance between original and updated models for a single MLB.

    This is an enhanced version of compare_model_performance that includes
    the improvement threshold decision logic for integrated validation.

    Args:
        original_model: Original trained model
        updated_model: Updated trained model
        X_val: Validation features
        y_val: Validation targets
        mlb: MLB identifier for logging
        improvement_threshold: Minimum improvement percentage to accept update

    Returns:
        Dictionary with performance comparison metrics and recommendation
    """
    logger = logging.getLogger(__name__)

    try:
        # Use existing compare_model_performance function
        metrics = compare_model_performance(
            original_model, updated_model, X_val, y_val, mlb
        )

        # Add recommendation based on improvement threshold
        improvement_pct = metrics.get("improvement_percentage", -100.0)

        if improvement_pct >= improvement_threshold:
            if improvement_pct > 0:
                recommendation = "accept_improved"
                metrics["recommendation"] = recommendation
                metrics["reason"] = f"Model improved by {improvement_pct:.2f}%"
            else:
                recommendation = "accept_maintained"
                metrics["recommendation"] = recommendation
                metrics["reason"] = (
                    f"Performance maintained within threshold ({improvement_pct:.2f}%)"
                )
        else:
            recommendation = "rollback"
            metrics["recommendation"] = recommendation
            metrics["reason"] = (
                f"Performance degraded by {improvement_pct:.2f}% (below {improvement_threshold}% threshold)"
            )

        logger.debug(
            f"MLB {mlb}: Performance comparison recommendation: {recommendation}"
        )
        return metrics

    except Exception as e:
        logger.error(f"MLB {mlb}: Error in single model performance comparison: {e}")
        return {
            "original_rmse": float("inf"),
            "updated_rmse": float("inf"),
            "original_mae": float("inf"),
            "updated_mae": float("inf"),
            "rmse_improvement_percentage": -100.0,
            "mae_improvement_percentage": -100.0,
            "improvement_percentage": -100.0,
            "validation_samples": 0,
            "recommendation": "rollback",
            "reason": f"Error during comparison: {e}",
            "error": str(e),
        }


def quick_model_health_check(model: object, mlb: str) -> bool:
    """
    Quick health check for a model to ensure it's usable for predictions.

    Args:
        model: Model to check
        mlb: MLB identifier for logging

    Returns:
        True if model appears healthy, False otherwise
    """
    logger = logging.getLogger(__name__)

    try:
        # Basic checks
        if not hasattr(model, "predict"):
            logger.warning(f"MLB {mlb}: Model missing predict method")
            return False

        # Try a dummy prediction to ensure model is functional
        if hasattr(model, "n_features_in_") and model.n_features_in_ > 0:
            dummy_features = np.zeros((1, model.n_features_in_))
            try:
                _ = model.predict(dummy_features)
                return True
            except Exception as e:
                logger.warning(f"MLB {mlb}: Model prediction test failed: {e}")
                return False
        else:
            # If we can't determine feature count, assume it's okay
            logger.debug(
                f"MLB {mlb}: Could not determine feature count, assuming model is healthy"
            )
            return True

    except Exception as e:
        logger.warning(f"MLB {mlb}: Model health check failed: {e}")
        return False
