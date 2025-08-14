"""
Test script for validation components in Step 3.2 of continuous learning implementation.
This script tests the forecast_validator and model_validator modules.
"""

import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor

# Add src path to import the modules
sys.path.append(str(Path(__file__).parent.parent))

from validation.forecast_validator import (
    validate_forecasts,
    calculate_historical_stats,
    validate_forecast_trends,
)
from validation.model_validator import (
    validate_daily_updates,
    validate_model_consistency,
    monitor_memory_usage_during_validation,
    compare_model_performance,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_test_forecast_data() -> Tuple[Dict[str, Tuple], Dict[str, Dict]]:
    """Create test forecast data for validation testing."""

    # Create test forecasts with various issues
    forecasts = {}

    # MLB1: Normal forecast
    future_dates = pd.date_range(start="2024-01-01", periods=90, freq="D")
    forecast_df1 = pd.DataFrame(
        {"prediction": np.random.randint(0, 50, 90)}, index=future_dates
    )
    forecasts["MLB1"] = (forecast_df1, "SKU001")

    # MLB2: Forecast with negative values
    forecast_df2 = pd.DataFrame(
        {
            "prediction": np.random.randint(-10, 20, 90)  # Some negative values
        },
        index=future_dates,
    )
    forecasts["MLB2"] = (forecast_df2, "SKU002")

    # MLB3: Forecast with NaN values
    predictions = np.random.randint(0, 30, 90).astype(float)
    predictions[5:10] = np.nan  # Insert some NaN values
    forecast_df3 = pd.DataFrame({"prediction": predictions}, index=future_dates)
    forecasts["MLB3"] = (forecast_df3, "SKU003")

    # MLB4: Forecast with extreme spikes
    predictions = np.random.randint(5, 15, 90).astype(float)
    predictions[20:25] = 1000  # Extreme spike
    forecast_df4 = pd.DataFrame({"prediction": predictions}, index=future_dates)
    forecasts["MLB4"] = (forecast_df4, "SKU004")

    # Create historical stats
    historical_stats = {
        "MLB1": {"avg_sales": 25, "max_sales": 80, "std_sales": 10, "median_sales": 22},
        "MLB2": {"avg_sales": 15, "max_sales": 45, "std_sales": 8, "median_sales": 12},
        "MLB3": {"avg_sales": 20, "max_sales": 60, "std_sales": 12, "median_sales": 18},
        "MLB4": {"avg_sales": 10, "max_sales": 35, "std_sales": 5, "median_sales": 8},
    }

    return forecasts, historical_stats


def create_test_model_data() -> Tuple[Dict, Dict, Dict]:
    """Create test model data for validation testing."""

    # Create simple XGBoost models
    original_models = {}
    updated_models = {}
    validation_data = {}

    # Create test data
    np.random.seed(42)  # For reproducible results

    for mlb in ["MLB1", "MLB2"]:
        # Create mock XGBoost models
        X_train = pd.DataFrame(
            {
                "feature1": np.random.rand(100),
                "feature2": np.random.rand(100),
                "feature3": np.random.rand(100),
                "feature4": np.random.rand(100),
            }
        )
        y_train = np.random.rand(100, 90)  # 90 outputs for 90-day forecast

        # Original model
        original_model = MultiOutputRegressor(
            xgb.XGBRegressor(n_estimators=10, random_state=42)
        )
        original_model.fit(X_train, y_train)
        original_models[mlb] = original_model

        # Updated model (slightly different)
        updated_model = MultiOutputRegressor(
            xgb.XGBRegressor(n_estimators=12, random_state=43)
        )
        updated_model.fit(X_train, y_train)
        updated_models[mlb] = updated_model

        # Validation data
        X_val = pd.DataFrame(
            {
                "feature1": np.random.rand(20),
                "feature2": np.random.rand(20),
                "feature3": np.random.rand(20),
                "feature4": np.random.rand(20),
            }
        )
        y_val = pd.DataFrame(np.random.rand(20, 90))
        validation_data[mlb] = (X_val, y_val)

    return original_models, updated_models, validation_data


def create_test_historical_data() -> pd.DataFrame:
    """Create test historical sales data."""

    # Create test data spanning multiple MLBs
    data = []
    dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")

    for mlb in ["MLB1", "MLB2", "MLB3", "MLB4"]:
        for date in dates:
            # Simulate realistic sales patterns with some randomness
            base_sales = {"MLB1": 25, "MLB2": 15, "MLB3": 20, "MLB4": 10}[mlb]
            sales = max(0, int(np.random.normal(base_sales, 5)))

            data.append({"mlb": mlb, "quant": sales, "date": date})

    return pd.DataFrame(data)


def test_forecast_validation():
    """Test forecast validation functions."""
    logger.info("=== Testing Forecast Validation ===")

    try:
        # Create test data
        forecasts, historical_stats = create_test_forecast_data()

        # Test validate_forecasts function
        logger.info("Testing validate_forecasts...")
        validated_forecasts, issues = validate_forecasts(forecasts, historical_stats)

        logger.info(f"Original forecasts: {len(forecasts)}")
        logger.info(f"Validated forecasts: {len(validated_forecasts)}")
        logger.info(f"Issues found: {len(issues)}")

        for issue in issues[:5]:  # Show first 5 issues
            logger.info(f"  - {issue}")

        # Test calculate_historical_stats function
        logger.info("Testing calculate_historical_stats...")
        historical_data = create_test_historical_data()
        calculated_stats = calculate_historical_stats(historical_data)

        logger.info(f"Historical stats calculated for {len(calculated_stats)} MLBs:")
        for mlb, stats in calculated_stats.items():
            logger.info(
                f"  {mlb}: avg={stats['avg_sales']:.1f}, max={stats['max_sales']:.1f}"
            )

        # Test validate_forecast_trends function
        logger.info("Testing validate_forecast_trends...")
        trend_warnings = validate_forecast_trends(validated_forecasts, historical_stats)

        logger.info(f"Trend warnings: {len(trend_warnings)}")
        for warning in trend_warnings:
            logger.info(f"  - {warning}")

        logger.info("✅ Forecast validation tests completed successfully")
        return True

    except Exception as e:
        logger.error(f"❌ Forecast validation test failed: {e}")
        return False


def test_model_validation():
    """Test model validation functions."""
    logger.info("=== Testing Model Validation ===")

    try:
        # Create test data
        original_models, updated_models, validation_data = create_test_model_data()

        # Test validate_daily_updates function
        logger.info("Testing validate_daily_updates...")
        validation_results = validate_daily_updates(
            original_models, updated_models, validation_data
        )

        logger.info("Validation results:")
        logger.info(f"  Total models: {validation_results['summary']['total_models']}")
        logger.info(
            f"  Improved models: {validation_results['summary']['improved_models']}"
        )
        logger.info(
            f"  Degraded models: {validation_results['summary']['degraded_models']}"
        )
        logger.info(
            f"  Rollback count: {validation_results['summary']['rollback_count']}"
        )

        # Test validate_model_consistency function
        logger.info("Testing validate_model_consistency...")
        is_consistent, consistency_issues = validate_model_consistency(updated_models)

        logger.info(f"Model consistency: {is_consistent}")
        if consistency_issues:
            logger.info("Consistency issues:")
            for issue in consistency_issues:
                logger.info(f"  - {issue}")

        # Test monitor_memory_usage_during_validation function
        logger.info("Testing monitor_memory_usage_during_validation...")
        memory_stats = monitor_memory_usage_during_validation()

        logger.info(
            f"Memory usage: {memory_stats['rss_mb']:.1f} MB RSS, "
            f"{memory_stats['percent']:.1f}% of system memory"
        )

        # Test compare_model_performance function
        logger.info("Testing compare_model_performance...")
        mlb = "MLB1"
        X_val, y_val = validation_data[mlb]
        performance_metrics = compare_model_performance(
            original_models[mlb], updated_models[mlb], X_val, y_val, mlb
        )

        logger.info(f"Performance comparison for {mlb}:")
        logger.info(
            f"  RMSE improvement: {performance_metrics['rmse_improvement_percentage']:.2f}%"
        )
        logger.info(
            f"  MAE improvement: {performance_metrics['mae_improvement_percentage']:.2f}%"
        )

        logger.info("✅ Model validation tests completed successfully")
        return True

    except Exception as e:
        logger.error(f"❌ Model validation test failed: {e}")
        return False


def main():
    """Main test function."""
    logger.info("Starting validation component tests...")

    forecast_test_passed = test_forecast_validation()
    model_test_passed = test_model_validation()

    if forecast_test_passed and model_test_passed:
        logger.info("🎉 All validation tests passed!")
        return True
    else:
        logger.error("❌ Some validation tests failed!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
