"""Function for fitting the machine learning model."""

import logging
import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
from pathlib import Path

from estimation.data_splitting import split_train_test
from config import AppConfig

from sklearn.multioutput import MultiOutputRegressor


def forecast_future_sales_direct(
    data: pd.DataFrame, forecast_days: int
) -> tuple[dict, dict]:
    """
    For each MLB in the data, perform a train/test split as usual,
    then train a multi-output XGBoost model using a direct forecasting approach:
    each training sample uses features at time t to predict the next forecast_days values.
    Finally, forecast future sales for the next forecast_days using the trained model.

    Args:
        data (pd.DataFrame): DataFrame containing data for multiple MLBs.
        forecast_days (int): Number of days into the future to forecast.

    Returns:
        tuple[dict, dict]: Tuple containing:
            - mlb_forecasts: Dictionary where each key is a MLB and the value is a tuple of (forecast_df, sku).
            - mlb_models: Dictionary where each key is a MLB and the value is the trained multi-output XGBoost model.
    """
    mlb_forecasts = {}
    mlb_models = {}
    FEATURES = ["day_of_week", "day_of_month", "rolling_mean_3", "lag_1"]

    # Debug logging: track filtering statistics
    total_mlbs = len(data["mlb"].unique())
    skipped_inactive = skipped_insufficient = skipped_training = processed = 0
    logging.info(f"Starting forecast for {total_mlbs} total MLBs")

    # Quick data freshness check
    if hasattr(data, "index") and isinstance(data.index, pd.DatetimeIndex):
        all_dates = data.index
    elif "date" in data.columns:
        all_dates = pd.to_datetime(data["date"])
    else:
        all_dates = None

    if all_dates is not None:
        global_max = all_dates.max()
        today = pd.Timestamp.now().normalize()
        days_old = (today - global_max).days
        logging.info(
            f"Processing data from {all_dates.min().date()} to {global_max.date()} ({days_old} days old)"
        )

    for mlb in data["mlb"].unique():
        mlb_data = data[data["mlb"] == mlb].copy()
        if not isinstance(mlb_data.index, pd.DatetimeIndex):
            mlb_data.index = pd.to_datetime(mlb_data.index)
        mlb_data = mlb_data.sort_index()

        # Get the SKU for this MLB (since each MLB maps to one SKU)
        sku = mlb_data["sku"].iloc[0] if "sku" in mlb_data.columns else None

        # Get date range for logging purposes
        last_date = mlb_data.index.max()

        # Skip MLBs with no recent data (inactive products)
        today = pd.Timestamp.now().normalize()
        cutoff_date = today - pd.Timedelta(days=AppConfig.ACTIVE_MLB_DAYS_THRESHOLD)

        if last_date < cutoff_date:
            skipped_inactive += 1
            days_since_last_data = (today - last_date).days
            logging.warning(
                f"MLB {mlb} (SKU {sku}) has no data in the last {AppConfig.ACTIVE_MLB_DAYS_THRESHOLD} days "
                f"(last data: {last_date.date()}, {days_since_last_data} days ago); "
                f"skipping forecast for inactive product."
            )
            continue

        # Ensure there are enough rows for multi-step forecasting:
        # Need at least forecast_days + sufficient training samples
        min_required_length = (
            forecast_days + 15
        )  # Need extra data for meaningful training
        if len(mlb_data) < min_required_length:
            skipped_insufficient += 1
            logging.warning(
                f"MLB {mlb} (SKU {sku}) has insufficient data ({len(mlb_data)} rows) for {forecast_days}-day "
                f"forecasting (requires at least {min_required_length} rows); skipping."
            )
            continue

        # Build the training set using a sliding window approach:
        X_train = []
        y_train = []
        # For each time index t where t + forecast_days exists, use features at time t
        # and target as quant values from t+1 to t+forecast_days.
        for i in range(len(mlb_data) - forecast_days):
            X_train.append(mlb_data.iloc[i][FEATURES].values)
            y_train.append(mlb_data.iloc[i + 1 : i + forecast_days + 1]["quant"].values)
        X_train = pd.DataFrame(X_train, columns=FEATURES)
        # Create target column names for each forecast step
        y_columns = [f"quant_{i+1}" for i in range(forecast_days)]
        y_train = pd.DataFrame(y_train, columns=y_columns)

        # Validate that we have sufficient training data
        if len(X_train) == 0 or len(y_train) == 0:
            skipped_training += 1
            logging.warning(
                f"MLB {mlb} (SKU {sku}) generated empty training set after processing; skipping forecast."
            )
            continue

        # Train multi-output regressor with XGBoost as the base estimator
        base_model = xgb.XGBRegressor(
            base_score=0.5,
            booster="gbtree",
            n_estimators=1000,
            objective="reg:squarederror",
            max_depth=3,
            learning_rate=0.01,
        )
        multi_model = MultiOutputRegressor(base_model)
        multi_model.fit(X_train, y_train)

        # Store the trained model for later use in continuous learning
        mlb_models[mlb] = multi_model

        # For forecasting, use the last available sample from training as input
        # Fix: Use the last row instead of potentially out-of-bounds index
        last_features = mlb_data.iloc[-1][FEATURES].values.reshape(1, -1)
        predictions = multi_model.predict(last_features)[0]

        # Round predictions to nearest integers (products sold in whole units)
        predictions = np.round(predictions).astype(int)

        # Build forecast DataFrame with future dates starting from tomorrow
        # Use current date instead of last_date to ensure predictions are always for the future
        today = pd.Timestamp.now().normalize()
        future_start = today + pd.Timedelta(days=1)
        future_dates = pd.date_range(
            start=future_start, periods=forecast_days, freq="D"
        )

        forecast_df = pd.DataFrame({"prediction": predictions}, index=future_dates)
        # Store forecast with MLB as key and include SKU for reference
        mlb_forecasts[mlb] = (forecast_df, sku)
        processed += 1

        # Log processing progress occasionally
        if processed % 50 == 0:
            logging.info(f"Processed {processed} MLBs so far...")

    # Final processing summary
    logging.info(
        f"Forecast Summary - Total: {total_mlbs}, "
        f"Skipped (inactive): {skipped_inactive}, "
        f"Skipped (insufficient data): {skipped_insufficient}, "
        f"Skipped (empty training): {skipped_training}, "
        f"Successfully processed: {processed}"
    )

    return mlb_forecasts, mlb_models


def forecast_future_sales_with_split(
    data: pd.DataFrame, forecast_days: int
) -> tuple[dict, dict]:
    """
    For each MLB in the data, perform a train/test split as usual,
    train an XGBoost model using the train and test sets, and then forecast
    future sales for the next forecast_days using the trained model.

    Args:
        data (pd.DataFrame): DataFrame containing data for multiple MLBs.
        forecast_days (int): Number of days into the future to forecast.

    Returns:
        tuple[dict, dict]: Tuple containing:
            - mlb_forecasts: Dictionary where each key is a MLB and the value is a tuple of (forecast_df, sku).
            - mlb_models: Dictionary where each key is a MLB and the value is the trained XGBoost model.
    """

    mlb_forecasts = {}
    mlb_models = {}
    for mlb in data["mlb"].unique():
        mlb_data = data[data["mlb"] == mlb].copy()

        if not isinstance(mlb_data.index, pd.DatetimeIndex):
            mlb_data.index = pd.to_datetime(mlb_data.index)

        # Get the SKU for this MLB (since each MLB maps to one SKU)
        sku = mlb_data["sku"].iloc[0] if "sku" in mlb_data.columns else None

        first_date = mlb_data.index.min()
        last_date = mlb_data.index.max()
        if (last_date - first_date) < pd.Timedelta(days=365):
            logging.warning(
                f"MLB {mlb} (SKU {sku}) has less than one year of data; skipping forecast."
            )
            continue

        # Skip MLBs with no recent data (inactive products)
        today = pd.Timestamp.now().normalize()
        cutoff_date = today - pd.Timedelta(days=AppConfig.ACTIVE_MLB_DAYS_THRESHOLD)
        if last_date < cutoff_date:
            days_since_last_data = (today - last_date).days
            logging.warning(
                f"MLB {mlb} (SKU {sku}) has no data in the last {AppConfig.ACTIVE_MLB_DAYS_THRESHOLD} days "
                f"(last data: {last_date.date()}, {days_since_last_data} days ago); "
                f"skipping forecast for inactive product."
            )
            continue

        # Split the MLB-specific data into train and test sets
        train, test = split_train_test(mlb_data)

        # Skip MLBs with insufficient training data
        if train.empty or len(train) < 10:  # Need minimum training samples
            logging.warning(
                f"MLB {mlb} (SKU {sku}) has insufficient training data ({len(train) if not train.empty else 0} samples); skipping forecast."
            )
            continue

        # Train the model using the train and test split
        model = train_xgboost_model(train, test)

        # Store the trained model for later use in continuous learning
        mlb_models[mlb] = model

        # Use current date for forecasting to ensure predictions are always for the future
        last_date = mlb_data.index.max()
        today = pd.Timestamp.now().normalize()
        future_start = today + pd.Timedelta(days=1)
        future_dates = pd.date_range(
            start=future_start, periods=forecast_days, freq="D"
        )

        target = "quant"
        # Initialize baseline values from the full mlb_data
        if len(mlb_data[target]) >= 3:
            last_values = list(mlb_data[target].iloc[-3:])
        else:
            last_values = [mlb_data[target].iloc[-1]] * 3

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
                }
            )

            pred = model.predict(features_future)[0]
            pred = np.round(pred).astype(int)  # Round to nearest integer
            predictions.append(pred)
            last_values.append(pred)

        forecast_df = pd.DataFrame({"prediction": predictions}, index=future_dates)
        # Store forecast with MLB as key and include SKU for reference
        mlb_forecasts[mlb] = (forecast_df, sku)

    return mlb_forecasts, mlb_models


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

    FEATURES = ["day_of_week", "day_of_month", "rolling_mean_3", "lag_1"]
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
