#!/usr/bin/env python3
"""
Test script for the incremental pipeline functionality.
Tests the incremental update functions and pipeline logic.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src path to import the modules
sys.path.append(str(Path(__file__).parent / "src" / "machine_learning"))

from estimation.model_forecast import (
    update_model_incremental,
    update_mlb_models_incremental,
    validate_model_improvement,
    _train_new_mlb_model,
)


def create_mock_data(n_samples: int = 200, mlb: str = "TEST_MLB") -> pd.DataFrame:
    """Create mock data for testing."""
    dates = pd.date_range(start="2024-01-01", periods=n_samples, freq="D")

    # Create realistic features
    np.random.seed(42)  # For reproducibility
    data = pd.DataFrame(
        {
            "mlb": [mlb] * n_samples,
            "sku": ["TEST_SKU"] * n_samples,
            "quant": np.random.poisson(lam=10, size=n_samples),  # Sales quantity
            "day_of_week": [d.dayofweek for d in dates],
            "day_of_month": [d.day for d in dates],
            "rolling_mean_3": np.random.normal(8, 2, n_samples),
            "lag_1": np.random.poisson(lam=9, size=n_samples),
        },
        index=dates,
    )

    return data


def test_train_new_model():
    """Test training a new model from scratch."""
    print("=== Testing New Model Training ===")

    try:
        # Create mock data
        mock_data = create_mock_data(150)

        # Train new model
        model = _train_new_mlb_model(mock_data, forecast_days=30)

        print("✅ Successfully trained new model")
        print(f"   Model type: {type(model)}")
        print(f"   Number of estimators: {len(model.estimators_)}")
        print(f"   First estimator type: {type(model.estimators_[0])}")

        # Test prediction
        FEATURES = ["day_of_week", "day_of_month", "rolling_mean_3", "lag_1"]
        test_features = mock_data.iloc[-1][FEATURES].values.reshape(1, -1)
        predictions = model.predict(test_features)

        print(f"   Prediction shape: {predictions.shape}")
        print(f"   Sample predictions: {predictions[0][:5]}")

        return True, model

    except Exception as e:
        print(f"❌ New model training failed: {e}")
        return False, None


def test_incremental_update():
    """Test incremental model update functionality."""
    print("\n=== Testing Incremental Model Update ===")

    try:
        # Create original training data
        original_data = create_mock_data(120, "TEST_MLB_1")
        print(f"Created original data: {len(original_data)} samples")

        # Train initial model
        original_model = _train_new_mlb_model(original_data, forecast_days=30)
        print("✅ Trained original model")

        # Create new data for incremental update
        new_data_start = original_data.index[-1] + pd.Timedelta(days=1)
        new_dates = pd.date_range(start=new_data_start, periods=50, freq="D")

        np.random.seed(43)  # Different seed for new data
        new_data = pd.DataFrame(
            {
                "mlb": ["TEST_MLB_1"] * 50,
                "sku": ["TEST_SKU"] * 50,
                "quant": np.random.poisson(
                    lam=12, size=50
                ),  # Slightly different pattern
                "day_of_week": [d.dayofweek for d in new_dates],
                "day_of_month": [d.day for d in new_dates],
                "rolling_mean_3": np.random.normal(10, 2, 50),
                "lag_1": np.random.poisson(lam=11, size=50),
            },
            index=new_dates,
        )

        print(f"Created new data: {len(new_data)} samples")

        # Test incremental update
        updated_model = update_model_incremental(
            original_model, new_data, additional_rounds=50, forecast_days=30
        )

        print("✅ Successfully performed incremental update")
        print(f"   Updated model type: {type(updated_model)}")
        print(f"   Number of estimators: {len(updated_model.estimators_)}")

        # Compare predictions
        FEATURES = ["day_of_week", "day_of_month", "rolling_mean_3", "lag_1"]
        test_features = new_data.iloc[-1][FEATURES].values.reshape(1, -1)

        original_pred = original_model.predict(test_features)
        updated_pred = updated_model.predict(test_features)

        print(f"   Original prediction sample: {original_pred[0][:3]}")
        print(f"   Updated prediction sample: {updated_pred[0][:3]}")

        # Check that predictions are different (model actually updated)
        diff = np.abs(original_pred - updated_pred).mean()
        print(f"   Average prediction difference: {diff:.3f}")

        if diff > 0.01:  # Some meaningful change
            print("✅ Model predictions changed after update (good)")
        else:
            print("⚠️ Model predictions barely changed - may need more training data")

        return True, original_model, updated_model

    except Exception as e:
        print(f"❌ Incremental update failed: {e}")
        import traceback

        traceback.print_exc()
        return False, None, None


def test_batch_update():
    """Test updating multiple MLB models."""
    print("\n=== Testing Batch MLB Model Update ===")

    try:
        # Create existing models for multiple MLBs
        mlbs = ["TEST_MLB_A", "TEST_MLB_B", "TEST_MLB_C"]
        existing_models = {}

        for mlb in mlbs:
            data = create_mock_data(200, mlb)
            model = _train_new_mlb_model(data, forecast_days=30)
            existing_models[mlb] = model

        print(f"✅ Created {len(existing_models)} existing models")

        # Create new data for all MLBs
        all_new_data = []
        for i, mlb in enumerate(mlbs):
            np.random.seed(50 + i)  # Different seed for each MLB
            new_data = create_mock_data(100, mlb)
            all_new_data.append(new_data)

        combined_new_data = pd.concat(all_new_data, ignore_index=True)
        print(
            f"Created combined new data: {len(combined_new_data)} samples for {len(mlbs)} MLBs"
        )

        # Test batch update
        updated_models = update_mlb_models_incremental(
            existing_models, combined_new_data, additional_rounds=30
        )

        print(f"✅ Successfully updated {len(updated_models)} models")

        # Verify all models were updated
        for mlb in mlbs:
            if mlb in updated_models:
                print(f"   ✅ {mlb}: Updated successfully")
            else:
                print(f"   ❌ {mlb}: Missing from updated models")

        return True

    except Exception as e:
        print(f"❌ Batch update failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_validation():
    """Test model validation functionality."""
    print("\n=== Testing Model Validation ===")

    try:
        # Create test data and models from previous test
        success, original_model, updated_model = test_incremental_update()
        if not success:
            print("❌ Cannot test validation - incremental update failed")
            return False

        # Create validation data
        val_data = create_mock_data(50, "TEST_VALIDATION")
        FEATURES = ["day_of_week", "day_of_month", "rolling_mean_3", "lag_1"]

        # Prepare validation data in the same format as training
        X_val = []
        y_val = []
        forecast_days = 30

        for i in range(len(val_data) - forecast_days):
            X_val.append(val_data.iloc[i][FEATURES].values)
            y_val.append(val_data.iloc[i + 1 : i + forecast_days + 1]["quant"].values)

        X_val = pd.DataFrame(X_val, columns=FEATURES)
        y_columns = [f"quant_{i+1}" for i in range(forecast_days)]
        y_val = pd.DataFrame(y_val, columns=y_columns)

        print(f"Created validation data: {len(X_val)} samples")

        # Test validation
        validation_results = validate_model_improvement(
            original_model, updated_model, (X_val, y_val)
        )

        print("✅ Validation completed")
        print(f"   Original RMSE: {validation_results['original_rmse']:.3f}")
        print(f"   Updated RMSE: {validation_results['updated_rmse']:.3f}")
        print(f"   Improvement: {validation_results['improvement_percentage']:.2f}%")
        print(f"   Is better: {validation_results['is_better']}")

        return True

    except Exception as e:
        print(f"❌ Validation test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests for incremental pipeline functionality."""
    print("Testing Step 2.2: Incremental Update Logic")
    print("=" * 60)

    tests_passed = 0
    total_tests = 4

    # Test 1: New model training
    if test_train_new_model()[0]:
        tests_passed += 1

    # Test 2: Incremental update
    if test_incremental_update()[0]:
        tests_passed += 1

    # Test 3: Batch update
    if test_batch_update():
        tests_passed += 1

    # Test 4: Validation
    if test_validation():
        tests_passed += 1

    print("\n" + "=" * 60)
    print(f"Tests completed: {tests_passed}/{total_tests} passed")

    if tests_passed == total_tests:
        print(
            "✅ All tests passed! Incremental update functionality is working correctly."
        )
    else:
        print(
            f"❌ {total_tests - tests_passed} tests failed. Please review the implementation."
        )

    print("\nNext steps:")
    print("- Test with real data using the incremental pipeline script")
    print("- Verify XGBoost continuation training is working as expected")
    print("- Test integration with the full pipeline workflow")


if __name__ == "__main__":
    main()
