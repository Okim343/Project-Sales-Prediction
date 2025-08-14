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
