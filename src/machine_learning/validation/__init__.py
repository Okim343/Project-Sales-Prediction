"""
Enhanced validation package for continuous learning pipeline.

This package provides validation components for Step 3.2 of the continuous learning implementation:
- forecast_validator: Validates forecast quality and detects anomalies
- model_validator: Validates model performance improvements and consistency
"""

from .forecast_validator import (
    validate_forecasts,
    calculate_historical_stats,
    validate_forecast_trends,
)

from .model_validator import (
    validate_daily_updates,
    validate_model_consistency,
    monitor_memory_usage_during_validation,
    compare_model_performance,
    validate_prediction_quality,
)

__version__ = "1.0.0"
__author__ = "Step 3.2 - Enhanced Validation Implementation"

__all__ = [
    "validate_forecasts",
    "calculate_historical_stats",
    "validate_forecast_trends",
    "validate_daily_updates",
    "validate_model_consistency",
    "monitor_memory_usage_during_validation",
    "compare_model_performance",
    "validate_prediction_quality",
]
