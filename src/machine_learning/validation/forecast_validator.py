"""Enhanced forecast validation for continuous learning pipeline."""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


def validate_forecasts(
    forecasts: Dict[str, Tuple], historical_stats: Dict[str, Dict]
) -> Tuple[Dict[str, Tuple], List[str]]:
    """
    Validate forecasts for quality control before saving.

    Args:
        forecasts: Dictionary where keys are MLB codes and values are tuples of (forecast_df, sku)
        historical_stats: Dictionary with historical statistics for each MLB
                         Format: {mlb: {'avg_sales': float, 'max_sales': float, 'std_sales': float}}

    Returns:
        Tuple containing:
            - valid_forecasts: Dictionary with validated forecasts that passed all checks
            - issues_list: List of validation issues encountered
    """
    logger = logging.getLogger(__name__)
    valid_forecasts = {}
    issues_list = []

    logger.info(f"Validating {len(forecasts)} forecasts...")

    for mlb, (forecast_df, sku) in forecasts.items():
        mlb_issues = []

        try:
            # Check 1: Negative predictions
            negative_count = (forecast_df["prediction"] < 0).sum()
            if negative_count > 0:
                mlb_issues.append(f"MLB {mlb}: {negative_count} negative predictions")
                # Set negative values to 0 for correction
                forecast_df["prediction"] = forecast_df["prediction"].clip(lower=0)

            # Check 2: NaN or infinite values
            nan_count = forecast_df["prediction"].isna().sum()
            inf_count = np.isinf(forecast_df["prediction"]).sum()

            if nan_count > 0:
                mlb_issues.append(f"MLB {mlb}: {nan_count} NaN predictions")
                # Replace NaN with 0
                forecast_df["prediction"] = forecast_df["prediction"].fillna(0)

            if inf_count > 0:
                mlb_issues.append(f"MLB {mlb}: {inf_count} infinite predictions")
                # Replace infinite values with historical average or 0
                hist_avg = historical_stats.get(mlb, {}).get("avg_sales", 0)
                forecast_df["prediction"] = forecast_df["prediction"].replace(
                    [np.inf, -np.inf], hist_avg
                )

            # Check 3: Values > 5x historical average (anomaly detection)
            if mlb in historical_stats and "avg_sales" in historical_stats[mlb]:
                hist_avg = historical_stats[mlb]["avg_sales"]
                threshold = max(hist_avg * 5, 10)  # At least 10 units threshold

                spike_count = (forecast_df["prediction"] > threshold).sum()
                if spike_count > 0:
                    max_prediction = forecast_df["prediction"].max()
                    mlb_issues.append(
                        f"MLB {mlb}: {spike_count} predictions > 5x historical average "
                        f"(max: {max_prediction:.1f}, threshold: {threshold:.1f})"
                    )

                    # Cap extreme values at threshold for correction
                    forecast_df["prediction"] = forecast_df["prediction"].clip(
                        upper=threshold
                    )

            # Check 4: Data type consistency
            if forecast_df["prediction"].dtype not in [
                np.int64,
                np.int32,
                np.float64,
                np.float32,
            ]:
                mlb_issues.append(
                    f"MLB {mlb}: Invalid data type {forecast_df['prediction'].dtype}"
                )
                forecast_df["prediction"] = pd.to_numeric(
                    forecast_df["prediction"], errors="coerce"
                ).fillna(0)

            # Check 5: Forecast length consistency
            expected_length = 90  # Default forecast days
            if len(forecast_df) != expected_length:
                mlb_issues.append(
                    f"MLB {mlb}: Incorrect forecast length {len(forecast_df)} (expected {expected_length})"
                )

            # If there were issues but they were corrected, keep the forecast with warning
            if mlb_issues:
                # Log issues but keep corrected forecast
                for issue in mlb_issues:
                    issues_list.append(issue)
                    logger.warning(issue)

                # Final validation after corrections
                final_nan_count = forecast_df["prediction"].isna().sum()
                final_inf_count = np.isinf(forecast_df["prediction"]).sum()

                if final_nan_count == 0 and final_inf_count == 0:
                    valid_forecasts[mlb] = (forecast_df, sku)
                    logger.info(f"MLB {mlb}: Issues corrected, forecast validated")
                else:
                    issues_list.append(
                        f"MLB {mlb}: Uncorrectable validation issues, forecast rejected"
                    )
                    logger.error(
                        f"MLB {mlb}: Forecast rejected due to uncorrectable issues"
                    )
            else:
                # No issues, forecast is valid
                valid_forecasts[mlb] = (forecast_df, sku)

        except Exception as e:
            issue_msg = f"MLB {mlb}: Validation failed with error: {e}"
            issues_list.append(issue_msg)
            logger.error(issue_msg)
            continue

    logger.info(
        f"Forecast validation completed: {len(valid_forecasts)}/{len(forecasts)} forecasts validated, "
        f"{len(issues_list)} issues found"
    )

    return valid_forecasts, issues_list


def calculate_historical_stats(data: pd.DataFrame) -> Dict[str, Dict]:
    """
    Calculate historical statistics for each MLB for validation reference.

    Args:
        data: Historical sales data with 'mlb' and 'quant' columns

    Returns:
        Dictionary with historical statistics for each MLB
        Format: {mlb: {'avg_sales': float, 'max_sales': float, 'std_sales': float, 'min_sales': float}}
    """
    logger = logging.getLogger(__name__)
    historical_stats = {}

    try:
        # Group by MLB and calculate statistics
        for mlb in data["mlb"].unique():
            mlb_data = data[data["mlb"] == mlb]
            sales_data = mlb_data["quant"]

            # Remove outliers for better statistics (using IQR method)
            Q1 = sales_data.quantile(0.25)
            Q3 = sales_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Filter out extreme outliers for statistics calculation
            clean_sales = sales_data[
                (sales_data >= lower_bound) & (sales_data <= upper_bound)
            ]

            if len(clean_sales) > 0:
                historical_stats[mlb] = {
                    "avg_sales": float(clean_sales.mean()),
                    "max_sales": float(
                        sales_data.max()
                    ),  # Use original max, not outlier-filtered
                    "min_sales": float(sales_data.min()),
                    "std_sales": float(clean_sales.std()),
                    "median_sales": float(clean_sales.median()),
                    "sample_count": len(sales_data),
                }
            else:
                # Fallback if no data after outlier removal
                historical_stats[mlb] = {
                    "avg_sales": float(sales_data.mean())
                    if len(sales_data) > 0
                    else 0.0,
                    "max_sales": float(sales_data.max())
                    if len(sales_data) > 0
                    else 0.0,
                    "min_sales": float(sales_data.min())
                    if len(sales_data) > 0
                    else 0.0,
                    "std_sales": float(sales_data.std())
                    if len(sales_data) > 0
                    else 0.0,
                    "median_sales": float(sales_data.median())
                    if len(sales_data) > 0
                    else 0.0,
                    "sample_count": len(sales_data),
                }

        logger.info(
            f"Calculated historical statistics for {len(historical_stats)} MLBs"
        )
        return historical_stats

    except Exception as e:
        logger.error(f"Failed to calculate historical statistics: {e}")
        return {}


def validate_forecast_trends(
    forecasts: Dict[str, Tuple], historical_stats: Dict[str, Dict]
) -> List[str]:
    """
    Validate forecast trends for reasonableness (optional advanced validation).

    Args:
        forecasts: Dictionary of validated forecasts
        historical_stats: Historical statistics for reference

    Returns:
        List of trend-related warnings
    """
    logger = logging.getLogger(__name__)
    trend_warnings = []

    try:
        for mlb, (forecast_df, sku) in forecasts.items():
            if mlb not in historical_stats:
                continue

            predictions = forecast_df["prediction"].values

            # Check for unrealistic trends
            # 1. Consistently increasing trend (may indicate overfitting)
            if len(predictions) > 7:  # Need at least a week of data
                weekly_means = [
                    predictions[i : i + 7].mean()
                    for i in range(0, len(predictions) - 6, 7)
                ]
                if len(weekly_means) > 1:
                    increasing_weeks = sum(
                        1
                        for i in range(1, len(weekly_means))
                        if weekly_means[i] > weekly_means[i - 1] * 1.1
                    )

                    if (
                        increasing_weeks == len(weekly_means) - 1
                    ):  # All weeks increasing by >10%
                        trend_warnings.append(
                            f"MLB {mlb}: Consistently increasing trend may indicate overfitting"
                        )

            # 2. High variance compared to historical data
            pred_std = predictions.std()
            hist_std = historical_stats[mlb].get("std_sales", 0)

            if hist_std > 0 and pred_std > hist_std * 3:
                trend_warnings.append(
                    f"MLB {mlb}: Forecast variance ({pred_std:.1f}) much higher than historical ({hist_std:.1f})"
                )

        if trend_warnings:
            logger.info(f"Found {len(trend_warnings)} trend warnings")

        return trend_warnings

    except Exception as e:
        logger.error(f"Failed to validate forecast trends: {e}")
        return [f"Trend validation failed: {e}"]


def validate_single_forecast(
    mlb: str,
    forecast_df: pd.DataFrame,
    sku: str,
    historical_stats: Dict[str, Dict],
    fix_issues: bool = True,
) -> Tuple[bool, Tuple, List[str]]:
    """
    Validate a single forecast for integrated validation flow.

    Args:
        mlb: MLB code
        forecast_df: Forecast DataFrame with 'prediction' column
        sku: SKU associated with the MLB
        historical_stats: Historical statistics for this MLB
        fix_issues: Whether to automatically fix correctable issues

    Returns:
        Tuple of (is_valid, forecast_tuple, issues_list)
        where forecast_tuple is (corrected_forecast_df, sku) if valid
    """
    logger = logging.getLogger(__name__)
    issues = []
    corrected_df = forecast_df.copy()

    try:
        # Use the existing batch validator for a single forecast
        single_forecast_dict = {mlb: (forecast_df, sku)}
        validated_dict, validation_issues = validate_forecasts(
            single_forecast_dict, historical_stats
        )

        # Check if forecast passed validation
        if mlb in validated_dict:
            corrected_forecast_df, corrected_sku = validated_dict[mlb]
            if validation_issues:
                issues.extend(validation_issues)
                if fix_issues:
                    logger.debug(
                        f"MLB {mlb}: Forecast had {len(validation_issues)} issues but was corrected"
                    )
                    return True, (corrected_forecast_df, corrected_sku), issues
                else:
                    logger.warning(
                        f"MLB {mlb}: Forecast has {len(validation_issues)} issues"
                    )
                    return False, (corrected_forecast_df, corrected_sku), issues
            else:
                logger.debug(f"MLB {mlb}: Forecast passed validation without issues")
                return True, (corrected_forecast_df, corrected_sku), issues
        else:
            # Forecast was rejected by validation
            rejection_reason = "Forecast failed validation and was rejected"
            if validation_issues:
                rejection_reason += f": {validation_issues[0]}"  # Include first issue
            issues.append(rejection_reason)
            logger.warning(f"MLB {mlb}: {rejection_reason}")
            return False, (corrected_df, sku), issues

    except Exception as e:
        error_msg = f"Single forecast validation failed: {e}"
        logger.error(f"MLB {mlb}: {error_msg}")
        issues.append(error_msg)
        return False, (corrected_df, sku), issues


def quick_forecast_sanity_check(
    mlb: str, forecast_df: pd.DataFrame, historical_stats: Dict[str, Dict]
) -> Tuple[bool, List[str]]:
    """
    Quick sanity check for a forecast to catch obvious issues early.

    Args:
        mlb: MLB code
        forecast_df: Forecast DataFrame
        historical_stats: Historical statistics

    Returns:
        Tuple of (passed_check, issues_list)
    """
    logger = logging.getLogger(__name__)
    issues = []

    try:
        predictions = forecast_df["prediction"].values

        # Check 1: No negative values
        if (predictions < 0).any():
            issues.append(f"Contains {(predictions < 0).sum()} negative predictions")

        # Check 2: No NaN or infinite values
        if np.isnan(predictions).any():
            issues.append(f"Contains {np.isnan(predictions).sum()} NaN values")

        if np.isinf(predictions).any():
            issues.append(f"Contains {np.isinf(predictions).sum()} infinite values")

        # Check 3: Reasonable range check using historical stats
        if mlb in historical_stats and "avg_sales" in historical_stats[mlb]:
            hist_avg = historical_stats[mlb]["avg_sales"]
            max_reasonable = hist_avg * 10  # 10x historical average threshold

            if (predictions > max_reasonable).any():
                spike_count = (predictions > max_reasonable).sum()
                max_pred = predictions.max()
                issues.append(
                    f"{spike_count} predictions exceed 10x historical average "
                    f"(max: {max_pred:.1f}, threshold: {max_reasonable:.1f})"
                )

        # Check 4: Data types
        if not np.issubdtype(predictions.dtype, np.number):
            issues.append(f"Non-numeric predictions detected: {predictions.dtype}")

        # Check 5: Expected length (assuming 90-day forecasts)
        expected_length = 90
        if len(predictions) != expected_length:
            issues.append(
                f"Unexpected forecast length: {len(predictions)} (expected {expected_length})"
            )

        passed = len(issues) == 0
        if not passed:
            logger.debug(f"MLB {mlb}: Quick sanity check found {len(issues)} issues")

        return passed, issues

    except Exception as e:
        error_msg = f"Quick sanity check failed: {e}"
        logger.error(f"MLB {mlb}: {error_msg}")
        return False, [error_msg]


def validate_forecast_quality_inline(
    mlb: str,
    forecast_df: pd.DataFrame,
    historical_stats: Dict[str, Dict],
    quality_threshold: float = 0.5,
) -> Tuple[bool, Dict]:
    """
    Validate forecast quality against historical patterns for integrated validation.

    Args:
        mlb: MLB code
        forecast_df: Forecast DataFrame
        historical_stats: Historical statistics
        quality_threshold: Minimum quality score (0-1) to pass

    Returns:
        Tuple of (passed_quality_check, quality_metrics)
    """
    logger = logging.getLogger(__name__)

    try:
        predictions = forecast_df["prediction"].values

        # Initialize quality metrics
        quality_metrics = {
            "mean_prediction": float(np.mean(predictions)),
            "std_prediction": float(np.std(predictions)),
            "quality_score": 0.0,
            "passed_threshold": False,
            "issues": [],
        }

        if mlb not in historical_stats:
            quality_metrics["issues"].append(
                "No historical stats available for quality assessment"
            )
            quality_metrics["quality_score"] = (
                quality_threshold  # Give benefit of doubt
            )
            quality_metrics["passed_threshold"] = True
            return True, quality_metrics

        hist_stats = historical_stats[mlb]
        hist_avg = hist_stats.get("avg_sales", 0)
        hist_std = hist_stats.get("std_sales", 0)

        # Quality metric 1: Mean similarity (30% weight)
        if hist_avg > 0:
            mean_similarity = 1 - min(
                1, abs(quality_metrics["mean_prediction"] - hist_avg) / hist_avg
            )
        else:
            mean_similarity = 0.5  # Neutral score if no historical average

        # Quality metric 2: Variance reasonableness (30% weight)
        if hist_std > 0:
            variance_ratio = quality_metrics["std_prediction"] / hist_std
            # Penalize if variance is too different (ideal range 0.5-2.0)
            if 0.5 <= variance_ratio <= 2.0:
                variance_score = 1.0
            elif variance_ratio < 0.5:
                variance_score = variance_ratio / 0.5  # Linear penalty below 0.5
            else:  # variance_ratio > 2.0
                variance_score = max(
                    0, 1 - (variance_ratio - 2.0) / 3.0
                )  # Linear penalty above 2.0
        else:
            variance_score = 0.5  # Neutral score

        # Quality metric 3: No extreme outliers (40% weight)
        outlier_threshold = max(hist_avg * 5, 10)  # 5x average or at least 10
        outlier_count = (predictions > outlier_threshold).sum()
        outlier_score = max(
            0, 1 - (outlier_count / len(predictions))
        )  # Penalty for outliers

        # Calculate overall quality score
        quality_score = (
            mean_similarity * 0.3 + variance_score * 0.3 + outlier_score * 0.4
        )
        quality_metrics["quality_score"] = float(quality_score)
        quality_metrics["passed_threshold"] = quality_score >= quality_threshold

        # Add detailed metrics
        quality_metrics.update(
            {
                "mean_similarity": float(mean_similarity),
                "variance_score": float(variance_score),
                "outlier_score": float(outlier_score),
                "outlier_count": int(outlier_count),
                "historical_avg": float(hist_avg),
                "historical_std": float(hist_std),
            }
        )

        # Log quality assessment
        if quality_metrics["passed_threshold"]:
            logger.debug(
                f"MLB {mlb}: Forecast quality score {quality_score:.3f} (passed)"
            )
        else:
            logger.warning(
                f"MLB {mlb}: Forecast quality score {quality_score:.3f} (failed threshold {quality_threshold})"
            )
            quality_metrics["issues"].append(
                f"Quality score {quality_score:.3f} below threshold {quality_threshold}"
            )

        return quality_metrics["passed_threshold"], quality_metrics

    except Exception as e:
        logger.error(f"MLB {mlb}: Forecast quality validation failed: {e}")
        return False, {
            "quality_score": 0.0,
            "passed_threshold": False,
            "error": str(e),
            "issues": [f"Quality validation error: {e}"],
        }


def correct_forecast_issues(
    mlb: str, forecast_df: pd.DataFrame, historical_stats: Dict[str, Dict]
) -> pd.DataFrame:
    """
    Automatically correct common forecast issues for integrated validation.

    Args:
        mlb: MLB code
        forecast_df: Original forecast DataFrame
        historical_stats: Historical statistics for corrections

    Returns:
        Corrected forecast DataFrame
    """
    logger = logging.getLogger(__name__)
    corrected_df = forecast_df.copy()

    try:
        predictions = corrected_df["prediction"].values

        # Correction 1: Fix negative values
        negative_count = (predictions < 0).sum()
        if negative_count > 0:
            corrected_df["prediction"] = corrected_df["prediction"].clip(lower=0)
            logger.debug(f"MLB {mlb}: Corrected {negative_count} negative predictions")

        # Correction 2: Fix NaN values
        nan_count = corrected_df["prediction"].isna().sum()
        if nan_count > 0:
            replacement_value = historical_stats.get(mlb, {}).get("avg_sales", 0)
            corrected_df["prediction"] = corrected_df["prediction"].fillna(
                replacement_value
            )
            logger.debug(
                f"MLB {mlb}: Corrected {nan_count} NaN predictions with {replacement_value}"
            )

        # Correction 3: Fix infinite values
        inf_mask = np.isinf(corrected_df["prediction"])
        inf_count = inf_mask.sum()
        if inf_count > 0:
            replacement_value = historical_stats.get(mlb, {}).get("avg_sales", 0)
            corrected_df.loc[inf_mask, "prediction"] = replacement_value
            logger.debug(
                f"MLB {mlb}: Corrected {inf_count} infinite predictions with {replacement_value}"
            )

        # Correction 4: Cap extreme outliers
        if mlb in historical_stats and "avg_sales" in historical_stats[mlb]:
            hist_avg = historical_stats[mlb]["avg_sales"]
            cap_threshold = max(hist_avg * 5, 10)  # 5x average or at least 10

            outlier_mask = corrected_df["prediction"] > cap_threshold
            outlier_count = outlier_mask.sum()
            if outlier_count > 0:
                corrected_df.loc[outlier_mask, "prediction"] = cap_threshold
                logger.debug(
                    f"MLB {mlb}: Capped {outlier_count} extreme outliers at {cap_threshold}"
                )

        # Correction 5: Ensure integer values for products
        corrected_df["prediction"] = corrected_df["prediction"].round().astype(int)

        return corrected_df

    except Exception as e:
        logger.error(f"MLB {mlb}: Failed to correct forecast issues: {e}")
        return forecast_df  # Return original if correction fails
