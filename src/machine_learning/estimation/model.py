"""Function for fitting the machine learning model."""

import logging
import pandas as pd
import xgboost as xgb
import pickle
from pathlib import Path

from estimation.data_splitting import split_train_test


def train_model_for_each_sku(data: pd.DataFrame) -> dict:
    """
    Train an XGBoost model for each SKU individually and save the predictions.

    This function assumes that the input DataFrame contains a column 'sku' indicating
    the SKU identifier. For each unique SKU, it splits the data into training and
    testing sets using the last month of data as the test set, trains the model, and
    collects predictions. SKUs with insufficient training data are skipped.

    Args:
        data (pd.DataFrame): DataFrame containing data for multiple SKUs.

    Returns:
        dict: A dictionary where keys are SKU values and values are the model predictions.
    """
    sku_predictions = {}
    # Iterate over each unique SKU
    for sku in data["sku"].unique():
        sku_data = data[data["sku"] == sku].copy()

        # Ensure the index is a DatetimeIndex before splitting
        if not isinstance(sku_data.index, pd.DatetimeIndex):
            sku_data.index = pd.to_datetime(sku_data.index)

        # Split the SKU-specific data into train and test sets
        train, test = split_train_test(sku_data)

        # Debugging statement
        # logging.info(f"SKU: {sku}, Train shape: {train.shape}, Test shape: {test.shape}")

        # Skip SKUs with insufficient training data
        if train.empty:
            logging.warning(f"SKU {sku} has insufficient data for training; skipping.")
            continue

        # Train the model for this SKU using the existing function
        model = train_xgboost_model(train, test)

        sku_predictions[sku] = model

    return sku_predictions


def save_regressors(regressors: dict, filepath: Path) -> None:
    """
    Save a dictionary of regressors to disk using pickle.

    Args:
        regressors (dict): Dictionary of regressors (or predictions) keyed by SKU.
        filepath (Path): The file path where the dictionary should be saved.
    """
    with filepath.open("wb") as f:
        pickle.dump(regressors, f)


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

    features = ["day_of_week", "day_of_month", "rolling_mean_3", "lag_1", "price"]
    target = "quant"

    x_train = train[features]
    y_train = train[target]

    x_test = test[features]
    y_test = test[target]

    regressor = xgb.XGBRegressor(
        base_score=0.5,
        booster="gbtree",
        n_estimators=1000,
        early_stopping_rounds=50,
        objective="reg:squarederror",
        max_depth=3,
        learning_rate=0.01,
    )

    regressor.fit(
        x_train,
        y_train,
        eval_set=[(x_train, y_train), (x_test, y_test)],
        verbose=100,
    )

    return regressor


def _fail_if_invalid_train_test_data(train: pd.DataFrame, test: pd.DataFrame):
    """Raise an error if train or test is not a DataFrame or is missing columns."""
    required_columns = {
        "day_of_week",
        "day_of_month",
        "rolling_mean_3",
        "lag_1",
        "quant",
    }

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


def _fail_if_train_test_empty(train: pd.DataFrame, test: pd.DataFrame):
    """Raise an error if train or test is empty."""
    for name, data in [("train", train), ("test", test)]:
        if data.empty:
            error_msg = f"'{name}' DataFrame is empty."
            raise ValueError(error_msg)


def _fail_if_target_contains_nan(train: pd.DataFrame, test: pd.DataFrame):
    """Raise an error if the target column 'quant' contains NaN values."""
    for name, data in [("train", train), ("test", test)]:
        if data["quant"].isna().any():
            error_msg = f"'quant' column in '{name}' contains NaN values."
            raise ValueError(error_msg)
