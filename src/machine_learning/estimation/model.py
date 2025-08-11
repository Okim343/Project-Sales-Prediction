"""Function for fitting the machine learning model."""

import logging
import pickle
from pathlib import Path
from typing import Dict

import pandas as pd
import xgboost as xgb

from estimation.data_splitting import split_train_test

# Model configuration constants
MODEL_FEATURES = ["day_of_week", "day_of_month", "rolling_mean_3", "lag_1", "price"]
TARGET_COLUMN = "quant"

# XGBoost hyperparameters
XGBOOST_PARAMS = {
    "base_score": 0.5,
    "booster": "gbtree",
    "n_estimators": 1000,
    "early_stopping_rounds": 50,
    "objective": "reg:squarederror",
    "max_depth": 3,
    "learning_rate": 0.01,
    "verbose": 100,
}

logger = logging.getLogger(__name__)


def train_model_for_each_sku(data: pd.DataFrame) -> Dict[str, xgb.XGBRegressor]:
    """
    Train an XGBoost model for each MLB individually and save the predictions.

    This function assumes that the input DataFrame contains a column 'mlb' indicating
    the MLB identifier. For each unique MLB, it splits the data into training and
    testing sets using the last month of data as the test set, trains the model, and
    collects predictions. MLBs with insufficient training data are skipped.

    Args:
        data (pd.DataFrame): DataFrame containing data for multiple MLBs.

    Returns:
        Dict[str, xgb.XGBRegressor]: A dictionary where keys are MLB values and values are trained models.
    """
    mlb_models = {}
    total_mlbs = len(data["mlb"].unique())

    # Iterate over each unique MLB
    for i, mlb in enumerate(data["mlb"].unique(), 1):
        # Get SKU for this MLB for logging purposes
        mlb_data = data[data["mlb"] == mlb].copy()
        sku = mlb_data["sku"].iloc[0] if "sku" in mlb_data.columns else "Unknown"

        logger.info(f"Training model for MLB {mlb} (SKU {sku}) ({i}/{total_mlbs})")

        # Ensure the index is a DatetimeIndex before splitting
        if not isinstance(mlb_data.index, pd.DatetimeIndex):
            mlb_data.index = pd.to_datetime(mlb_data.index)

        # Split the MLB-specific data into train and test sets
        train, test = split_train_test(mlb_data)

        # Skip MLBs with insufficient training data
        if train.empty:
            logger.warning(
                f"MLB {mlb} (SKU {sku}) has insufficient data for training; skipping."
            )
            continue

        try:
            # Train the model for this MLB
            model = train_xgboost_model(train, test)
            mlb_models[mlb] = model
            logger.info(f"Successfully trained model for MLB {mlb} (SKU {sku})")
        except Exception as e:
            logger.error(f"Failed to train model for MLB {mlb} (SKU {sku}): {e}")
            continue

    logger.info(f"Training complete. Successfully trained {len(mlb_models)} models.")
    return mlb_models


def save_regressors(regressors: Dict[str, xgb.XGBRegressor], filepath: Path) -> None:
    """
    Save a dictionary of regressors to disk using pickle.

    Args:
        regressors: Dictionary of trained regressors keyed by MLB.
        filepath: The file path where the dictionary should be saved.
    """
    try:
        with filepath.open("wb") as f:
            pickle.dump(regressors, f)
        logger.info(f"Successfully saved {len(regressors)} regressors to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save regressors to {filepath}: {e}")
        raise


def train_xgboost_model(train: pd.DataFrame, test: pd.DataFrame) -> xgb.XGBRegressor:
    """Train an XGBoost model and return the trained regressor.

    Args:
        train (pd.DataFrame): Training dataset with pre-created features and target.
        test (pd.DataFrame): Testing dataset with pre-created features and target.

    Returns:
        xgb.XGBRegressor: Fitted XGBoost model predictions.
    """
    _fail_if_invalid_train_test_data(train, test)
    _fail_if_train_test_empty(train, test)
    _fail_if_target_contains_nan(train, test)

    x_train = train[MODEL_FEATURES]
    y_train = train[TARGET_COLUMN]

    x_test = test[MODEL_FEATURES]
    y_test = test[TARGET_COLUMN]

    # Create regressor with predefined parameters
    regressor = xgb.XGBRegressor(
        **{
            k: v
            for k, v in XGBOOST_PARAMS.items()
            if k not in ["verbose", "early_stopping_rounds"]
        }
    )

    # Train the model
    regressor.fit(
        x_train,
        y_train,
        eval_set=[(x_train, y_train), (x_test, y_test)],
        early_stopping_rounds=XGBOOST_PARAMS["early_stopping_rounds"],
        verbose=XGBOOST_PARAMS["verbose"],
    )

    return regressor


def _fail_if_invalid_train_test_data(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Raise an error if train or test is not a DataFrame or is missing columns."""
    required_columns = set(MODEL_FEATURES + [TARGET_COLUMN])

    for name, data in [("train", train), ("test", test)]:
        if not isinstance(data, pd.DataFrame):
            error_msg = f"'{name}' must be a pandas DataFrame, got {type(data)}."
            raise TypeError(error_msg)

        missing_columns = required_columns - set(data.columns)
        if missing_columns:
            error_msg = (
                f"'{name}' DataFrame is missing required columns: {missing_columns}"
            )
            raise ValueError(error_msg)


def _fail_if_train_test_empty(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Raise an error if train or test is empty."""
    for name, data in [("train", train), ("test", test)]:
        if data.empty:
            error_msg = f"'{name}' DataFrame is empty."
            raise ValueError(error_msg)


def _fail_if_target_contains_nan(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Raise an error if the target column contains NaN values."""
    for name, data in [("train", train), ("test", test)]:
        if data[TARGET_COLUMN].isna().any():
            error_msg = f"'{TARGET_COLUMN}' column in '{name}' contains NaN values."
            raise ValueError(error_msg)
