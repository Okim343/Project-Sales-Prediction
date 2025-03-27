"""Function for fitting the machine learning model."""

import logging
import pandas as pd
import xgboost as xgb
import pickle
from pathlib import Path

from estimation.data_splitting import split_train_test


def forecast_future_sales_with_split(data: pd.DataFrame, forecast_days: int) -> dict:
    """
    For each SKU in the data, perform a train/test split as usual,
    train an XGBoost model using the train and test sets, and then forecast
    future sales for the next forecast_days using the trained model.

    Args:
        data (pd.DataFrame): DataFrame containing data for multiple SKUs.
        forecast_days (int): Number of days into the future to forecast.

    Returns:
        dict: Dictionary where each key is a SKU and the value is a DataFrame of forecasts.
    """

    sku_forecasts = {}
    for sku in data["sku"].unique():
        sku_data = data[data["sku"] == sku].copy()

        if not isinstance(sku_data.index, pd.DatetimeIndex):
            sku_data.index = pd.to_datetime(sku_data.index)

        first_date = sku_data.index.min()
        last_date = sku_data.index.max()
        if (last_date - first_date) < pd.Timedelta(days=365):
            logging.warning(
                f"SKU {sku} has less than one year of data; skipping forecast."
            )
            continue

        # Split the SKU-specific data into train and test sets
        train, test = split_train_test(sku_data)

        # Skip SKUs with insufficient training data
        if train.empty:
            logging.warning(
                f"SKU {sku} has insufficient data for training; skipping forecast."
            )
            continue

        # Train the model using the train and test split
        model = train_xgboost_model(train, test)

        # Use the full SKU data for forecasting baseline (last available actual data)
        last_date = sku_data.index.max()
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1), periods=forecast_days, freq="D"
        )

        target = "quant"
        # Initialize baseline values from the full sku_data
        if len(sku_data[target]) >= 3:
            last_values = list(sku_data[target].iloc[-3:])
        else:
            last_values = [sku_data[target].iloc[-1]] * 3

        last_price_per_unit = sku_data["price_per_unit"].iloc[-1]

        predictions = []
        for date in future_dates:
            day_of_week = date.dayofweek
            day_of_month = date.day
            lag_1 = last_values[-1]
            rolling_mean_3 = sum(last_values[-3:]) / 3

            features_future = pd.DataFrame(
                {
                    "day_of_week": [day_of_week],
                    "day_of_month": [day_of_month],
                    "rolling_mean_3": [rolling_mean_3],
                    "lag_1": [lag_1],
                    "price_per_unit": [last_price_per_unit],
                }
            )

            pred = model.predict(features_future)[0]
            predictions.append(pred)
            last_values.append(pred)

        forecast_df = pd.DataFrame({"prediction": predictions}, index=future_dates)
        sku_forecasts[sku] = forecast_df

    return sku_forecasts


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

    FEATURES = [
        "day_of_week",
        "day_of_month",
        "rolling_mean_3",
        "lag_1",
        "price_per_unit",
    ]
    target = "quant"

    x_train = train[FEATURES]
    y_train = train[target]

    x_test = test[FEATURES]
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


def _set_quant_as_int(data: pd.DataFrame):
    """Converts the 'quant' column to integer format."""
    data = data.copy()
    data["quant"] = data["quant"].round(0).astype(int)
    return data


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
