"""
Comprehensive tests for the model comparison module.

This module tests the model version comparison functionality used in monthly retraining
to ensure systematic comparison of old vs new models with proper recommendations.
"""

import sys
import unittest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb
from pathlib import Path

# Add src path to import the modules
sys.path.append(str(Path(__file__).parent.parent))

# Import the modules we're testing
from validation.model_comparison import (
    compare_model_versions,
    compare_single_model_version,
    calculate_performance_metrics,
    make_update_recommendation,
    generate_comparison_report,
    recommend_model_updates,
)


class TestModelComparison(unittest.TestCase):
    """Test cases for model comparison functionality."""

    def setUp(self):
        """Set up test fixtures with sample data and models."""
        # Create sample validation data
        np.random.seed(42)  # For reproducible tests
        self.sample_size = 50
        self.forecast_days = 3  # Smaller for testing

        # Create features
        self.X_val = pd.DataFrame(
            {
                "day_of_week": np.random.randint(0, 7, self.sample_size),
                "day_of_month": np.random.randint(1, 31, self.sample_size),
                "rolling_mean_3": np.random.uniform(1, 10, self.sample_size),
                "lag_1": np.random.uniform(1, 10, self.sample_size),
            }
        )

        # Create targets (multi-output for forecast_days)
        y_columns = [f"quant_{i+1}" for i in range(self.forecast_days)]
        self.y_val = pd.DataFrame(
            {col: np.random.uniform(1, 15, self.sample_size) for col in y_columns}
        )

        # Create mock models that return predictable results
        self.mock_model_good = Mock()
        self.mock_model_bad = Mock()
        self.mock_model_identical = Mock()

        # Set up predictable prediction patterns
        # Good model: predicts close to actual values (RMSE ~1.0)
        good_predictions = self.y_val.values + np.random.normal(0, 1, self.y_val.shape)
        self.mock_model_good.predict.return_value = good_predictions

        # Bad model: predicts far from actual values (RMSE ~5.0)
        bad_predictions = self.y_val.values + np.random.normal(0, 5, self.y_val.shape)
        self.mock_model_bad.predict.return_value = bad_predictions

        # Identical model: same predictions as good model
        self.mock_model_identical.predict.return_value = good_predictions

        # Sample validation data dictionary
        self.validation_data = {
            "MLB001": (self.X_val, self.y_val),
            "MLB002": (self.X_val[:30], self.y_val[:30]),  # Different size
        }

    def test_calculate_performance_metrics_basic(self):
        """Test basic performance metrics calculation."""
        metrics = calculate_performance_metrics(
            self.mock_model_good, self.X_val, self.y_val, "test_model"
        )

        # Check that all expected metrics are present
        expected_keys = [
            "rmse",
            "mae",
            "mean_prediction",
            "mean_actual",
            "prediction_std",
            "actual_std",
            "correlation",
            "sample_count",
        ]
        for key in expected_keys:
            self.assertIn(key, metrics)
            self.assertIsInstance(metrics[key], (int, float))
            self.assertFalse(np.isnan(metrics[key]) or np.isinf(metrics[key]))

        # Check reasonable value ranges
        self.assertGreater(metrics["rmse"], 0)
        self.assertGreater(metrics["mae"], 0)
        self.assertEqual(
            metrics["sample_count"], len(self.y_val) * len(self.y_val.columns)
        )

    def test_calculate_performance_metrics_error_handling(self):
        """Test error handling in performance metrics calculation."""
        # Create a model that raises an exception
        error_model = Mock()
        error_model.predict.side_effect = ValueError("Model prediction failed")

        metrics = calculate_performance_metrics(
            error_model, self.X_val, self.y_val, "error_model"
        )

        # Should return error metrics
        self.assertEqual(metrics["rmse"], float("inf"))
        self.assertEqual(metrics["mae"], float("inf"))
        self.assertIn("error", metrics)

    def test_compare_single_model_version_improvement(self):
        """Test single model comparison when new model is better."""
        result = compare_single_model_version(
            old_model=self.mock_model_bad,
            new_model=self.mock_model_good,
            X_val=self.X_val,
            y_val=self.y_val,
            mlb="MLB001",
            improvement_threshold=-5.0,
        )

        # Check structure
        expected_keys = [
            "old_model_metrics",
            "new_model_metrics",
            "rmse_improvement_percentage",
            "mae_improvement_percentage",
            "improvement_percentage",
            "recommendation",
            "reason",
            "validation_samples",
        ]
        for key in expected_keys:
            self.assertIn(key, result)

        # New model should be better (positive improvement)
        self.assertGreater(result["improvement_percentage"], 0)
        self.assertEqual(result["recommendation"], "update")
        self.assertIn("improvement", result["reason"].lower())

    def test_compare_single_model_version_degradation(self):
        """Test single model comparison when new model is worse."""
        result = compare_single_model_version(
            old_model=self.mock_model_good,
            new_model=self.mock_model_bad,
            X_val=self.X_val,
            y_val=self.y_val,
            mlb="MLB001",
            improvement_threshold=-5.0,
        )

        # New model should be worse (negative improvement)
        self.assertLess(result["improvement_percentage"], 0)
        # Should recommend keeping old if degradation is significant
        if result["improvement_percentage"] < -5.0:
            self.assertEqual(result["recommendation"], "keep_old")

    def test_compare_single_model_version_identical(self):
        """Test single model comparison when models perform identically."""
        result = compare_single_model_version(
            old_model=self.mock_model_good,
            new_model=self.mock_model_identical,
            X_val=self.X_val,
            y_val=self.y_val,
            mlb="MLB001",
            improvement_threshold=-5.0,
        )

        # Should have very small improvement difference (near zero)
        self.assertAlmostEqual(result["improvement_percentage"], 0, places=1)

    def test_make_update_recommendation_clear_improvement(self):
        """Test recommendation logic for clear improvement."""
        recommendation, reason = make_update_recommendation(
            improvement_percentage=10.0,
            improvement_threshold=-5.0,
            statistical_significance=None,
            mlb="MLB001",
        )

        self.assertEqual(recommendation, "update")
        self.assertIn("improvement", reason.lower())

    def test_make_update_recommendation_severe_degradation(self):
        """Test recommendation logic for severe degradation."""
        recommendation, reason = make_update_recommendation(
            improvement_percentage=-25.0,
            improvement_threshold=-5.0,
            statistical_significance=None,
            mlb="MLB001",
        )

        self.assertEqual(recommendation, "keep_old")
        self.assertIn("severe", reason.lower())

    def test_make_update_recommendation_marginal_change(self):
        """Test recommendation logic for marginal changes."""
        # Small improvement within threshold
        recommendation, reason = make_update_recommendation(
            improvement_percentage=2.0,
            improvement_threshold=-5.0,
            statistical_significance=None,
            mlb="MLB001",
        )

        self.assertEqual(recommendation, "update")

        # Small degradation within threshold
        recommendation, reason = make_update_recommendation(
            improvement_percentage=-2.0,
            improvement_threshold=-5.0,
            statistical_significance=None,
            mlb="MLB001",
        )

        self.assertEqual(recommendation, "update")

    def test_compare_model_versions_basic(self):
        """Test basic model version comparison functionality."""
        old_models = {
            "MLB001": self.mock_model_bad,
            "MLB002": self.mock_model_bad,
        }
        new_models = {
            "MLB001": self.mock_model_good,
            "MLB002": self.mock_model_good,
            "MLB003": self.mock_model_good,  # New MLB
        }

        results = compare_model_versions(
            old_models=old_models,
            new_models=new_models,
            validation_data=self.validation_data,
            improvement_threshold=-5.0,
        )

        # Check structure
        expected_keys = [
            "comparison_results",
            "recommendations",
            "summary",
            "memory_usage",
        ]
        for key in expected_keys:
            self.assertIn(key, results)

        # Check summary statistics
        summary = results["summary"]
        self.assertGreater(summary["total_compared"], 0)
        self.assertGreaterEqual(summary["recommend_update"], 0)

        # New MLB should always be recommended for update
        self.assertEqual(results["recommendations"].get("MLB003"), "update")

    def test_compare_model_versions_no_validation_data(self):
        """Test handling of MLBs without validation data."""
        old_models = {"MLB999": self.mock_model_good}
        new_models = {"MLB999": self.mock_model_good}
        validation_data = {}  # No validation data

        results = compare_model_versions(
            old_models=old_models,
            new_models=new_models,
            validation_data=validation_data,
        )

        # Should handle gracefully
        self.assertIn("comparison_results", results)
        self.assertIn("recommendations", results)
        # MLB999 should get recommendation based on fallback logic
        self.assertIn("MLB999", results["recommendations"])

    def test_generate_comparison_report(self):
        """Test comparison report generation."""
        # Create sample comparison results
        sample_results = {
            "summary": {
                "total_compared": 5,
                "recommend_update": 3,
                "recommend_keep_old": 1,
                "uncertain": 1,
                "comparison_errors": 0,
                "avg_improvement_pct": 2.5,
            },
            "memory_usage": {
                "initial_mb": 100.0,
                "final_mb": 120.0,
                "peak_increase_mb": 20.0,
            },
            "comparison_results": {
                "MLB001": {"improvement_percentage": 15.0, "recommendation": "update"},
                "MLB002": {
                    "improvement_percentage": -12.0,
                    "recommendation": "keep_old",
                },
            },
        }

        report = generate_comparison_report(sample_results)

        # Check that report contains key information
        self.assertIn("MODEL VERSION COMPARISON", report)
        self.assertIn("Total comparisons", report)
        self.assertIn("Recommended updates", report)
        self.assertIn("Memory Usage", report)
        self.assertIn("MLB001", report)  # Should show significant changes

    def test_recommend_model_updates_conservative(self):
        """Test model update recommendations in conservative mode."""
        sample_results = {
            "recommendations": {
                "MLB001": "update",
                "MLB002": "keep_old",
                "MLB003": "uncertain",
            },
            "comparison_results": {
                "MLB001": {"improvement_percentage": 10.0},
                "MLB002": {"improvement_percentage": -8.0},
                "MLB003": {"improvement_percentage": 1.0},
            },
        }

        models_to_update, reasoning = recommend_model_updates(
            sample_results, conservative_mode=True
        )

        # In conservative mode, uncertain should not be updated
        self.assertTrue(models_to_update.get("MLB001", False))
        self.assertFalse(models_to_update.get("MLB002", False))
        self.assertFalse(models_to_update.get("MLB003", False))  # Conservative mode

        # Check reasoning
        self.assertIsInstance(reasoning, list)
        self.assertGreater(len(reasoning), 0)

    def test_recommend_model_updates_normal(self):
        """Test model update recommendations in normal mode."""
        sample_results = {
            "recommendations": {
                "MLB001": "update",
                "MLB002": "keep_old",
                "MLB003": "uncertain",
            },
            "comparison_results": {
                "MLB001": {"improvement_percentage": 10.0},
                "MLB002": {"improvement_percentage": -8.0},
                "MLB003": {"improvement_percentage": 1.0},
            },
        }

        models_to_update, reasoning = recommend_model_updates(
            sample_results, conservative_mode=False
        )

        # In normal mode, uncertain should default to update
        self.assertTrue(models_to_update.get("MLB001", False))
        self.assertFalse(models_to_update.get("MLB002", False))
        self.assertTrue(
            models_to_update.get("MLB003", False)
        )  # Normal mode defaults to update

    @patch("validation.model_comparison.psutil.Process")
    def test_memory_monitoring(self, mock_process):
        """Test memory usage monitoring during comparison."""
        # Mock memory info
        mock_memory_info = Mock()
        mock_memory_info.rss = 100 * 1024 * 1024  # 100 MB
        mock_process.return_value.memory_info.return_value = mock_memory_info

        old_models = {"MLB001": self.mock_model_good}
        new_models = {"MLB001": self.mock_model_good}

        results = compare_model_versions(
            old_models=old_models,
            new_models=new_models,
            validation_data=self.validation_data,
        )

        # Should have memory usage information
        self.assertIn("memory_usage", results)
        memory_info = results["memory_usage"]
        self.assertIn("initial_mb", memory_info)
        self.assertIn("final_mb", memory_info)
        self.assertIn("peak_increase_mb", memory_info)

    def test_statistical_significance_calculation(self):
        """Test statistical significance calculation in model comparison."""
        # This test verifies that statistical significance is calculated when enough samples
        result = compare_single_model_version(
            old_model=self.mock_model_bad,
            new_model=self.mock_model_good,
            X_val=self.X_val,
            y_val=self.y_val,
            mlb="MLB001",
            significance_level=0.05,
        )

        # Should have statistical significance information
        self.assertIn("statistical_significance", result)
        if result["statistical_significance"] is not None:
            stat_sig = result["statistical_significance"]
            if "error" not in stat_sig:
                self.assertIn("t_statistic", stat_sig)
                self.assertIn("p_value", stat_sig)
                self.assertIn("is_significant", stat_sig)

    def test_edge_case_empty_models(self):
        """Test handling of edge case with empty model dictionaries."""
        results = compare_model_versions(
            old_models={},
            new_models={},
            validation_data={},
        )

        # Should handle gracefully
        self.assertIn("summary", results)
        self.assertEqual(results["summary"]["total_compared"], 0)

    def test_edge_case_single_sample(self):
        """Test handling of edge case with very small validation data."""
        small_X = self.X_val.iloc[:1]  # Single sample
        small_y = self.y_val.iloc[:1]

        result = compare_single_model_version(
            old_model=self.mock_model_good,
            new_model=self.mock_model_good,
            X_val=small_X,
            y_val=small_y,
            mlb="MLB001",
        )

        # Should complete without error
        self.assertIn("validation_samples", result)
        self.assertEqual(result["validation_samples"], 1)


class TestModelComparisonIntegration(unittest.TestCase):
    """Integration tests for model comparison with real XGBoost models."""

    def setUp(self):
        """Set up integration test fixtures with real models."""
        np.random.seed(42)

        # Create more realistic training data
        n_samples = 100
        self.X_train = pd.DataFrame(
            {
                "day_of_week": np.random.randint(0, 7, n_samples),
                "day_of_month": np.random.randint(1, 31, n_samples),
                "rolling_mean_3": np.random.uniform(1, 10, n_samples),
                "lag_1": np.random.uniform(1, 10, n_samples),
            }
        )

        # Create target with some real relationship to features
        self.y_train = (
            self.X_train["rolling_mean_3"] * 2 + np.random.normal(0, 1, n_samples) + 5
        )

        # Create validation data
        n_val = 30
        self.X_val = pd.DataFrame(
            {
                "day_of_week": np.random.randint(0, 7, n_val),
                "day_of_month": np.random.randint(1, 31, n_val),
                "rolling_mean_3": np.random.uniform(1, 10, n_val),
                "lag_1": np.random.uniform(1, 10, n_val),
            }
        )

        self.y_val = pd.DataFrame(
            {
                "quant_1": self.X_val["rolling_mean_3"] * 2
                + np.random.normal(0, 1, n_val)
                + 5,
            }
        )

    def test_real_xgboost_model_comparison(self):
        """Test comparison with actual XGBoost models."""
        try:
            # Train two XGBoost models with different parameters
            model1 = xgb.XGBRegressor(
                n_estimators=10, max_depth=2, learning_rate=0.1, random_state=42
            )
            model2 = xgb.XGBRegressor(
                n_estimators=20, max_depth=3, learning_rate=0.05, random_state=42
            )

            model1.fit(self.X_train, self.y_train)
            model2.fit(self.X_train, self.y_train)

            # Compare models
            result = compare_single_model_version(
                old_model=model1,
                new_model=model2,
                X_val=self.X_val,
                y_val=self.y_val,
                mlb="MLB_REAL",
            )

            # Should complete successfully
            self.assertIn("recommendation", result)
            self.assertIn("improvement_percentage", result)
            self.assertIsInstance(result["improvement_percentage"], (int, float))

        except ImportError:
            self.skipTest("XGBoost not available for integration testing")

    def test_multioutput_regressor_comparison(self):
        """Test comparison with MultiOutputRegressor models."""
        try:
            # Create multi-output target
            y_multi = pd.DataFrame(
                {
                    "quant_1": self.y_train,
                    "quant_2": self.y_train
                    + np.random.normal(0, 0.5, len(self.y_train)),
                    "quant_3": self.y_train
                    + np.random.normal(0, 0.5, len(self.y_train)),
                }
            )

            # Train MultiOutputRegressor models
            base_model1 = xgb.XGBRegressor(
                n_estimators=10, max_depth=2, random_state=42
            )
            base_model2 = xgb.XGBRegressor(
                n_estimators=15, max_depth=3, random_state=42
            )

            model1 = MultiOutputRegressor(base_model1)
            model2 = MultiOutputRegressor(base_model2)

            model1.fit(self.X_train, y_multi)
            model2.fit(self.X_train, y_multi)

            # Create corresponding validation targets
            y_val_multi = pd.DataFrame(
                {
                    "quant_1": self.y_val["quant_1"],
                    "quant_2": self.y_val["quant_1"]
                    + np.random.normal(0, 0.5, len(self.y_val)),
                    "quant_3": self.y_val["quant_1"]
                    + np.random.normal(0, 0.5, len(self.y_val)),
                }
            )

            # Compare models
            result = compare_single_model_version(
                old_model=model1,
                new_model=model2,
                X_val=self.X_val,
                y_val=y_val_multi,
                mlb="MLB_MULTI",
            )

            # Should handle multi-output models correctly
            self.assertIn("recommendation", result)
            self.assertIn("improvement_percentage", result)

        except ImportError:
            self.skipTest("XGBoost not available for integration testing")


if __name__ == "__main__":
    # Run all tests
    unittest.main(verbosity=2)
