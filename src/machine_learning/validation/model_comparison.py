"""
Model version comparison module for continuous learning pipeline.

This module provides systematic model version comparison following existing validation architecture.
Used primarily for monthly full retrain mode to compare new models against existing ones.
"""

import logging
import numpy as np
import pandas as pd
import psutil
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import stats

logger = logging.getLogger(__name__)


def compare_model_versions(
    old_models: Dict[str, object],
    new_models: Dict[str, object],
    validation_data: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]],
    improvement_threshold: float = -5.0,
    significance_level: float = 0.05,
) -> Dict[str, Dict]:
    """
    Compare performance between old and new model versions for multiple MLBs.

    Args:
        old_models: Dictionary of existing models keyed by MLB
        new_models: Dictionary of new models keyed by MLB
        validation_data: Dictionary of validation data by MLB
                        Format: {mlb: (X_val, y_val)}
        improvement_threshold: Minimum improvement percentage to recommend update
        significance_level: Statistical significance level for performance tests

    Returns:
        Dictionary with comparison results compatible with existing validation metrics format
        Format: {
            'comparison_results': {mlb: {...}},
            'recommendations': {mlb: str},  # 'update', 'keep_old', 'uncertain'
            'summary': {...},
            'memory_usage': {...}
        }
    """
    logger.info(f"Starting model version comparison for {len(new_models)} MLBs...")

    # Monitor memory usage during comparison
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB

    results = {
        "comparison_results": {},
        "recommendations": {},
        "summary": {
            "total_compared": 0,
            "recommend_update": 0,
            "recommend_keep_old": 0,
            "uncertain": 0,
            "comparison_errors": 0,
            "avg_improvement_pct": 0.0,
        },
        "memory_usage": {},
    }

    # Find MLBs that exist in both old and new models
    common_mlbs = set(old_models.keys()).intersection(set(new_models.keys()))

    # Also include MLBs that only exist in new models (new products)
    new_only_mlbs = set(new_models.keys()) - set(old_models.keys())

    logger.info(
        f"Found {len(common_mlbs)} MLBs for comparison, {len(new_only_mlbs)} new MLBs"
    )

    # Track improvements for summary statistics
    improvements = []

    # Compare existing MLBs
    for mlb in common_mlbs:
        if mlb not in validation_data:
            logger.warning(
                f"MLB {mlb}: No validation data available, skipping comparison"
            )
            # Provide fallback recommendation for MLBs without validation data
            results["comparison_results"][mlb] = {
                "recommendation": "keep_old",
                "reason": "No validation data available - defaulting to keep existing model for safety",
                "validation_samples": 0,
                "no_validation_data": True,
            }
            results["recommendations"][mlb] = "keep_old"
            results["summary"]["recommend_keep_old"] += 1
            continue

        try:
            old_model = old_models[mlb]
            new_model = new_models[mlb]
            X_val, y_val = validation_data[mlb]

            # Perform detailed comparison
            comparison_result = compare_single_model_version(
                old_model,
                new_model,
                X_val,
                y_val,
                mlb,
                improvement_threshold,
                significance_level,
            )

            results["comparison_results"][mlb] = comparison_result
            results["recommendations"][mlb] = comparison_result["recommendation"]

            # Track improvement for summary
            improvement_pct = comparison_result.get("improvement_percentage", 0.0)
            improvements.append(improvement_pct)

            # Update summary counters
            results["summary"]["total_compared"] += 1

            if comparison_result["recommendation"] == "update":
                results["summary"]["recommend_update"] += 1
            elif comparison_result["recommendation"] == "keep_old":
                results["summary"]["recommend_keep_old"] += 1
            else:
                results["summary"]["uncertain"] += 1

        except Exception as e:
            logger.error(f"MLB {mlb}: Model comparison failed: {e}")
            results["summary"]["comparison_errors"] += 1
            results["recommendations"][mlb] = "keep_old"  # Safe fallback

    # Handle new MLBs (always recommend update since no old model exists)
    for mlb in new_only_mlbs:
        results["recommendations"][mlb] = "update"
        results["comparison_results"][mlb] = {
            "recommendation": "update",
            "reason": "New MLB - no existing model to compare against",
            "is_new_mlb": True,
        }
        results["summary"]["recommend_update"] += 1

    # Calculate summary statistics
    if improvements:
        results["summary"]["avg_improvement_pct"] = float(np.mean(improvements))

    # Record memory usage
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    results["memory_usage"] = {
        "initial_mb": round(initial_memory, 1),
        "final_mb": round(final_memory, 1),
        "peak_increase_mb": round(final_memory - initial_memory, 1),
    }

    # Log summary
    summary = results["summary"]
    logger.info("=== MODEL VERSION COMPARISON SUMMARY ===")
    logger.info(f"Total comparisons: {summary['total_compared']}")
    logger.info(f"Recommend update: {summary['recommend_update']}")
    logger.info(f"Recommend keep old: {summary['recommend_keep_old']}")
    logger.info(f"Uncertain: {summary['uncertain']}")
    logger.info(f"Comparison errors: {summary['comparison_errors']}")
    logger.info(f"Average improvement: {summary['avg_improvement_pct']:.2f}%")
    logger.info(
        f"Memory usage: {results['memory_usage']['peak_increase_mb']} MB increase"
    )

    return results


def compare_single_model_version(
    old_model: object,
    new_model: object,
    X_val: pd.DataFrame,
    y_val: pd.DataFrame,
    mlb: str,
    improvement_threshold: float = -5.0,
    significance_level: float = 0.05,
) -> Dict[str, any]:
    """
    Compare performance between old and new model for a single MLB.

    Args:
        old_model: Existing model
        new_model: New model to compare
        X_val: Validation features
        y_val: Validation targets
        mlb: MLB identifier for logging
        improvement_threshold: Minimum improvement percentage to recommend update
        significance_level: Statistical significance level for performance tests

    Returns:
        Dictionary with detailed comparison metrics and recommendation
    """
    logger.debug(f"MLB {mlb}: Performing detailed model version comparison...")

    try:
        # Calculate performance metrics for both models
        old_metrics = calculate_performance_metrics(
            old_model, X_val, y_val, f"{mlb}_old"
        )
        new_metrics = calculate_performance_metrics(
            new_model, X_val, y_val, f"{mlb}_new"
        )

        # Calculate improvement percentages
        rmse_improvement = (
            ((old_metrics["rmse"] - new_metrics["rmse"]) / old_metrics["rmse"]) * 100
            if old_metrics["rmse"] > 0
            else 0
        )
        mae_improvement = (
            ((old_metrics["mae"] - new_metrics["mae"]) / old_metrics["mae"]) * 100
            if old_metrics["mae"] > 0
            else 0
        )

        # Use RMSE as primary metric for decisions
        primary_improvement = rmse_improvement

        # Perform statistical significance test if enough samples
        statistical_significance = None
        if len(X_val) >= 10:  # Minimum samples for meaningful test
            try:
                # Generate predictions for statistical test
                old_pred = old_model.predict(X_val)
                new_pred = new_model.predict(X_val)

                # Handle multi-output predictions
                if len(old_pred.shape) > 1 and old_pred.shape[1] > 1:
                    old_pred = old_pred.flatten()
                    new_pred = new_pred.flatten()
                    y_val_flat = y_val.values.flatten()
                else:
                    if len(old_pred.shape) > 1:
                        old_pred = old_pred.flatten()
                    if len(new_pred.shape) > 1:
                        new_pred = new_pred.flatten()
                    y_val_flat = y_val.values.flatten()

                # Calculate residuals
                old_residuals = np.abs(y_val_flat - old_pred)
                new_residuals = np.abs(y_val_flat - new_pred)

                # Paired t-test on residuals (lower is better)
                t_stat, p_value = stats.ttest_rel(old_residuals, new_residuals)
                statistical_significance = {
                    "t_statistic": float(t_stat),
                    "p_value": float(p_value),
                    "is_significant": p_value < significance_level,
                    "new_model_better": t_stat > 0 and p_value < significance_level,
                }

            except Exception as stat_error:
                logger.debug(f"MLB {mlb}: Statistical test failed: {stat_error}")
                statistical_significance = {"error": str(stat_error)}

        # Make recommendation based on improvement threshold and statistical significance
        recommendation, reason = make_update_recommendation(
            primary_improvement, improvement_threshold, statistical_significance, mlb
        )

        # Compile results
        result = {
            "old_model_metrics": old_metrics,
            "new_model_metrics": new_metrics,
            "rmse_improvement_percentage": float(rmse_improvement),
            "mae_improvement_percentage": float(mae_improvement),
            "improvement_percentage": float(primary_improvement),
            "statistical_significance": statistical_significance,
            "recommendation": recommendation,
            "reason": reason,
            "validation_samples": len(X_val),
            "improvement_threshold_used": improvement_threshold,
        }

        logger.debug(
            f"MLB {mlb}: RMSE {old_metrics['rmse']:.2f} -> {new_metrics['rmse']:.2f} "
            f"({rmse_improvement:+.2f}%), Recommendation: {recommendation}"
        )

        return result

    except Exception as e:
        logger.error(f"MLB {mlb}: Error in model version comparison: {e}")
        return {
            "old_model_metrics": {"rmse": float("inf"), "mae": float("inf")},
            "new_model_metrics": {"rmse": float("inf"), "mae": float("inf")},
            "rmse_improvement_percentage": -100.0,
            "mae_improvement_percentage": -100.0,
            "improvement_percentage": -100.0,
            "statistical_significance": None,
            "recommendation": "keep_old",
            "reason": f"Comparison failed: {e}",
            "validation_samples": 0,
            "error": str(e),
        }


def calculate_performance_metrics(
    model: object, X_val: pd.DataFrame, y_val: pd.DataFrame, model_id: str
) -> Dict[str, float]:
    """
    Calculate performance metrics for a single model.
    Reuses existing metric calculation patterns from model_validator.py.

    Args:
        model: Trained model to evaluate
        X_val: Validation features
        y_val: Validation targets
        model_id: Model identifier for logging

    Returns:
        Dictionary with performance metrics
    """
    try:
        # Generate predictions
        predictions = model.predict(X_val)

        # Handle multi-output predictions
        if len(predictions.shape) > 1 and predictions.shape[1] > 1:
            # Multi-output case - flatten for metric calculation
            predictions = predictions.flatten()
            y_val_flat = y_val.values.flatten()
        else:
            # Single output case
            if len(predictions.shape) > 1:
                predictions = predictions.flatten()
            y_val_flat = y_val.values.flatten()

        # Calculate core metrics
        rmse = float(np.sqrt(mean_squared_error(y_val_flat, predictions)))
        mae = float(mean_absolute_error(y_val_flat, predictions))

        # Calculate additional quality metrics
        mean_prediction = float(np.mean(predictions))
        mean_actual = float(np.mean(y_val_flat))
        prediction_std = float(np.std(predictions))
        actual_std = float(np.std(y_val_flat))

        # Calculate correlation if possible
        correlation = 0.0
        if len(predictions) > 1 and prediction_std > 0 and actual_std > 0:
            try:
                correlation = float(np.corrcoef(predictions, y_val_flat)[0, 1])
                if np.isnan(correlation):
                    correlation = 0.0
            except (ValueError, IndexError, TypeError):
                correlation = 0.0

        return {
            "rmse": rmse,
            "mae": mae,
            "mean_prediction": mean_prediction,
            "mean_actual": mean_actual,
            "prediction_std": prediction_std,
            "actual_std": actual_std,
            "correlation": correlation,
            "sample_count": len(y_val_flat),
        }

    except Exception as e:
        logger.error(f"Error calculating metrics for {model_id}: {e}")
        return {
            "rmse": float("inf"),
            "mae": float("inf"),
            "mean_prediction": 0.0,
            "mean_actual": 0.0,
            "prediction_std": 0.0,
            "actual_std": 0.0,
            "correlation": 0.0,
            "sample_count": 0,
            "error": str(e),
        }


def make_update_recommendation(
    improvement_percentage: float,
    improvement_threshold: float,
    statistical_significance: Optional[Dict],
    mlb: str,
) -> Tuple[str, str]:
    """
    Make recommendation on whether to update to new model based on performance and significance.

    Args:
        improvement_percentage: Primary improvement percentage (positive = better)
        improvement_threshold: Minimum improvement threshold (can be negative)
        statistical_significance: Statistical test results or None
        mlb: MLB identifier for logging

    Returns:
        Tuple of (recommendation, reason) where recommendation is 'update', 'keep_old', or 'uncertain'
    """
    # Primary decision based on improvement threshold
    meets_threshold = improvement_percentage >= improvement_threshold

    # Consider statistical significance if available
    is_statistically_significant = False
    if statistical_significance and "is_significant" in statistical_significance:
        is_statistically_significant = statistical_significance[
            "is_significant"
        ] and statistical_significance.get("new_model_better", False)

    # Decision logic
    if improvement_percentage > 5.0:  # Clear improvement
        if is_statistically_significant or statistical_significance is None:
            return "update", f"Significant improvement of {improvement_percentage:.2f}%"
        else:
            return (
                "update",
                f"Large improvement of {improvement_percentage:.2f}% (not statistically significant but practical)",
            )

    elif meets_threshold:  # Meets minimum threshold
        if improvement_percentage > 0:
            if is_statistically_significant or statistical_significance is None:
                return "update", f"Model improved by {improvement_percentage:.2f}%"
            else:
                return (
                    "uncertain",
                    f"Improvement of {improvement_percentage:.2f}% not statistically significant",
                )
        else:
            return (
                "update",
                f"Performance maintained within acceptable threshold ({improvement_percentage:.2f}%)",
            )

    else:  # Below threshold
        if improvement_percentage < -20.0:  # Severe degradation
            return (
                "keep_old",
                f"Severe performance degradation of {improvement_percentage:.2f}%",
            )
        else:
            return (
                "keep_old",
                f"Performance degraded by {improvement_percentage:.2f}% (below {improvement_threshold}% threshold)",
            )


def generate_comparison_report(comparison_results: Dict) -> str:
    """
    Generate a detailed comparison report for logging and analysis.

    Args:
        comparison_results: Results from compare_model_versions()

    Returns:
        Formatted report string
    """
    if not comparison_results or "summary" not in comparison_results:
        return "No comparison results available"

    summary = comparison_results["summary"]
    memory = comparison_results.get("memory_usage", {})

    report_lines = [
        "=" * 60,
        "MODEL VERSION COMPARISON DETAILED REPORT",
        "=" * 60,
        f"Total comparisons performed: {summary.get('total_compared', 0)}",
        f"Recommended updates: {summary.get('recommend_update', 0)}",
        f"Recommended to keep old: {summary.get('recommend_keep_old', 0)}",
        f"Uncertain recommendations: {summary.get('uncertain', 0)}",
        f"Comparison errors: {summary.get('comparison_errors', 0)}",
        f"Average improvement: {summary.get('avg_improvement_pct', 0):.2f}%",
        "",
        "Memory Usage:",
        f"  Initial: {memory.get('initial_mb', 0):.1f} MB",
        f"  Final: {memory.get('final_mb', 0):.1f} MB",
        f"  Increase: {memory.get('peak_increase_mb', 0):.1f} MB",
        "",
    ]

    # Add details for MLBs with significant changes
    if "comparison_results" in comparison_results:
        significant_changes = []
        for mlb, result in comparison_results["comparison_results"].items():
            improvement = result.get("improvement_percentage", 0)
            if abs(improvement) > 10.0:  # Show changes > 10%
                significant_changes.append(
                    (mlb, improvement, result.get("recommendation", "unknown"))
                )

        if significant_changes:
            report_lines.append(
                "Significant Changes (>10% improvement or degradation):"
            )
            for mlb, improvement, recommendation in sorted(
                significant_changes, key=lambda x: x[1], reverse=True
            ):
                report_lines.append(f"  {mlb}: {improvement:+.1f}% ({recommendation})")
            report_lines.append("")

    report_lines.append("=" * 60)

    return "\n".join(report_lines)


def recommend_model_updates(
    comparison_results: Dict, conservative_mode: bool = False
) -> Tuple[Dict[str, object], List[str]]:
    """
    Generate final recommendations for which models to update based on comparison results.

    Args:
        comparison_results: Results from compare_model_versions()
        conservative_mode: If True, only recommend updates for clearly better models

    Returns:
        Tuple of (models_to_update_dict, reasoning_list)
    """
    recommendations = comparison_results.get("recommendations", {})
    detailed_results = comparison_results.get("comparison_results", {})

    models_to_update = {}
    reasoning = []

    # Count recommendations by type
    update_count = sum(1 for rec in recommendations.values() if rec == "update")
    keep_old_count = sum(1 for rec in recommendations.values() if rec == "keep_old")
    uncertain_count = sum(1 for rec in recommendations.values() if rec == "uncertain")

    reasoning.append(
        f"Recommendation summary: {update_count} update, {keep_old_count} keep old, {uncertain_count} uncertain"
    )

    for mlb, recommendation in recommendations.items():
        detailed = detailed_results.get(mlb, {})
        improvement = detailed.get("improvement_percentage", 0)

        if recommendation == "update":
            models_to_update[mlb] = True
            if detailed.get("is_new_mlb", False):
                reasoning.append(f"MLB {mlb}: Update (new MLB)")
            else:
                reasoning.append(f"MLB {mlb}: Update ({improvement:+.1f}% improvement)")

        elif recommendation == "uncertain":
            if conservative_mode:
                reasoning.append(f"MLB {mlb}: Keep old (conservative mode, uncertain)")
            else:
                models_to_update[mlb] = True
                reasoning.append(f"MLB {mlb}: Update (uncertain but defaulting to new)")

        else:  # keep_old
            reasoning.append(f"MLB {mlb}: Keep old ({improvement:+.1f}% change)")

    return models_to_update, reasoning
