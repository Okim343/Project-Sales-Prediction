"""Function for fitting the machine learning model."""

import logging
import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import mean_squared_error

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
    logger = logging.getLogger(__name__)
    mlb_forecasts = {}
    mlb_models = {}
    FEATURES = ["day_of_week", "day_of_month", "rolling_mean_3", "lag_1"]

    # Debug logging: track filtering statistics
    total_mlbs = len(data["mlb"].unique())
    skipped_inactive = skipped_insufficient = skipped_training = processed = 0
    logger.info(f"Starting forecast for {total_mlbs} total MLBs")

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
        logger.info(
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
            logger.warning(
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
            logger.warning(
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
            logger.warning(
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
            logger.info(f"Processed {processed} MLBs so far...")

    # Final processing summary
    logger.info(
        f"Forecast Summary - Total: {total_mlbs}, "
        f"Skipped (inactive): {skipped_inactive}, "
        f"Skipped (insufficient data): {skipped_insufficient}, "
        f"Skipped (empty training): {skipped_training}, "
        f"Successfully processed: {processed}"
    )

    return mlb_forecasts, mlb_models


def forecast_future_sales_direct_limited(
    data: pd.DataFrame, forecast_days: int, max_mlbs: int
) -> tuple[dict, dict]:
    """
    Limited version of forecast_future_sales_direct that stops after processing max_mlbs successful forecasts.
    This is used for testing to speed up the pipeline.

    Args:
        data (pd.DataFrame): DataFrame containing data for multiple MLBs.
        forecast_days (int): Number of days into the future to forecast.
        max_mlbs (int): Maximum number of MLBs to successfully process before stopping.

    Returns:
        tuple[dict, dict]: Tuple containing:
            - mlb_forecasts: Dictionary where each key is a MLB and the value is a tuple of (forecast_df, sku).
            - mlb_models: Dictionary where each key is a MLB and the value is the trained multi-output XGBoost model.
    """
    logger = logging.getLogger(__name__)
    mlb_forecasts = {}
    mlb_models = {}
    FEATURES = ["day_of_week", "day_of_month", "rolling_mean_3", "lag_1"]

    # Debug logging: track filtering statistics
    total_mlbs = len(data["mlb"].unique())
    skipped_inactive = skipped_insufficient = skipped_training = processed = 0
    logger.info(
        f"Starting limited forecast for up to {max_mlbs} MLBs out of {total_mlbs} total MLBs"
    )

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
        logger.info(
            f"Processing data from {all_dates.min().date()} to {global_max.date()} ({days_old} days old)"
        )

    for mlb in data["mlb"].unique():
        # Stop if we've reached the maximum number of successful forecasts
        if processed >= max_mlbs:
            logger.info(
                f"Reached maximum MLB limit ({max_mlbs}). Stopping forecast processing."
            )
            break

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
            logger.warning(
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
            logger.warning(
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
            logger.warning(
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

        # Log processing progress for test runs
        logger.info(
            f"Successfully processed MLB {mlb} (SKU {sku}) - {processed}/{max_mlbs} completed"
        )

    # Final processing summary
    remaining_mlbs = (
        total_mlbs
        - skipped_inactive
        - skipped_insufficient
        - skipped_training
        - processed
    )
    logger.info(
        f"Limited Forecast Summary - Total: {total_mlbs}, "
        f"Skipped (inactive): {skipped_inactive}, "
        f"Skipped (insufficient data): {skipped_insufficient}, "
        f"Skipped (empty training): {skipped_training}, "
        f"Successfully processed: {processed}/{max_mlbs} (limit reached), "
        f"Remaining viable MLBs: {remaining_mlbs}"
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
    logger = logging.getLogger(__name__)
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
            logger.warning(
                f"MLB {mlb} (SKU {sku}) has less than one year of data; skipping forecast."
            )
            continue

        # Skip MLBs with no recent data (inactive products)
        today = pd.Timestamp.now().normalize()
        cutoff_date = today - pd.Timedelta(days=AppConfig.ACTIVE_MLB_DAYS_THRESHOLD)
        if last_date < cutoff_date:
            days_since_last_data = (today - last_date).days
            logger.warning(
                f"MLB {mlb} (SKU {sku}) has no data in the last {AppConfig.ACTIVE_MLB_DAYS_THRESHOLD} days "
                f"(last data: {last_date.date()}, {days_since_last_data} days ago); "
                f"skipping forecast for inactive product."
            )
            continue

        # Split the MLB-specific data into train and test sets
        train, test = split_train_test(mlb_data)

        # Skip MLBs with insufficient training data
        if train.empty or len(train) < 10:  # Need minimum training samples
            logger.warning(
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


# ============================================================================
# Incremental Update Functions for Continuous Learning
# ============================================================================


def update_model_incremental(
    existing_model: MultiOutputRegressor,
    new_data: pd.DataFrame,
    target_column: str = "quant",
    feature_columns: Optional[List[str]] = None,
    additional_rounds: int = 100,
    forecast_days: int = 90,
) -> MultiOutputRegressor:
    """
    Update an existing XGBoost model with new data using continuation training.

    This function implements XGBoost's continuation training feature to incrementally
    update models with new data without retraining from scratch.

    Args:
        existing_model: Previously trained MultiOutputRegressor with XGBRegressor base estimators
        new_data: DataFrame with new training data for incremental update
        target_column: Name of target variable column (default: "quant")
        feature_columns: List of feature column names (default: AppConfig.MODEL_FEATURES)
        additional_rounds: Number of additional boosting rounds for continuation (default: 100)
        forecast_days: Number of forecast days for multi-output structure (default: 90)

    Returns:
        Updated MultiOutputRegressor model with continued training

    Raises:
        ValueError: If feature consistency checks fail or data is insufficient
        TypeError: If existing_model is not a MultiOutputRegressor
    """
    logger = logging.getLogger(__name__)

    # Validate inputs
    if not isinstance(existing_model, MultiOutputRegressor):
        raise TypeError("existing_model must be a MultiOutputRegressor")

    if feature_columns is None:
        # Use the core features that don't include 'price' to maintain compatibility
        feature_columns = ["day_of_week", "day_of_month", "rolling_mean_3", "lag_1"]

    if len(new_data) < forecast_days + 5:  # Need enough data for meaningful training
        raise ValueError(
            f"Insufficient new data: need at least {forecast_days + 5} rows, got {len(new_data)}"
        )

    try:
        # Ensure feature consistency - critical for XGBoost continuation
        if not all(col in new_data.columns for col in feature_columns):
            missing_cols = [
                col for col in feature_columns if col not in new_data.columns
            ]
            raise ValueError(f"New data missing required features: {missing_cols}")

        # Prepare training data using same sliding window approach as original training
        X_new = []
        y_new = []

        for i in range(len(new_data) - forecast_days):
            X_new.append(new_data.iloc[i][feature_columns].values)
            y_new.append(
                new_data.iloc[i + 1 : i + forecast_days + 1][target_column].values
            )

        if len(X_new) == 0:
            raise ValueError("Generated empty training set from new data")

        X_new = pd.DataFrame(X_new, columns=feature_columns)
        y_columns = [f"{target_column}_{i+1}" for i in range(forecast_days)]
        y_new = pd.DataFrame(y_new, columns=y_columns)

        logger.info(f"Prepared incremental training data: {len(X_new)} samples")

        # Get original model hyperparameters from first estimator
        base_estimator = existing_model.estimators_[0]

        # Create new base models for continuation with reduced rounds
        new_base_models = []
        for i in range(len(existing_model.estimators_)):
            new_base_model = xgb.XGBRegressor(
                n_estimators=additional_rounds,
                max_depth=base_estimator.max_depth,
                learning_rate=base_estimator.learning_rate,
                base_score=base_estimator.base_score,
                booster=base_estimator.booster,
                objective=base_estimator.objective,
                random_state=base_estimator.random_state,
            )
            new_base_models.append(new_base_model)

        # Create updated multi-output regressor
        updated_model = MultiOutputRegressor(
            new_base_models[0]
        )  # Will be replaced below
        updated_model.estimators_ = []

        # Continue training for each output (forecast day)
        for i, (new_base_model, existing_estimator) in enumerate(
            zip(new_base_models, existing_model.estimators_)
        ):
            try:
                # Continue training using existing model as base
                new_base_model.fit(
                    X_new,
                    y_new.iloc[:, i],  # Target for this forecast day
                    xgb_model=existing_estimator,  # Critical: use existing model for continuation
                )
                updated_model.estimators_.append(new_base_model)

            except Exception as e:
                logger.warning(f"Continuation training failed for output {i}: {e}")
                # Fallback: train from scratch for this output
                fallback_model = xgb.XGBRegressor(
                    n_estimators=additional_rounds,
                    max_depth=base_estimator.max_depth,
                    learning_rate=base_estimator.learning_rate,
                )
                fallback_model.fit(X_new, y_new.iloc[:, i])
                updated_model.estimators_.append(fallback_model)

        # Validate the updated model
        total_rounds = sum(
            est.get_booster().num_boosted_rounds() for est in updated_model.estimators_
        )
        avg_rounds = total_rounds / len(updated_model.estimators_)
        logger.info(
            f"Incremental update completed. Average boosted rounds: {avg_rounds:.1f}"
        )

        return updated_model

    except Exception as e:
        logger.error(f"Incremental model update failed: {e}")
        raise


def update_mlb_models_incremental(
    existing_models: Dict[str, MultiOutputRegressor],
    new_data: pd.DataFrame,
    additional_rounds: int = 100,
) -> Dict[str, MultiOutputRegressor]:
    """
    Update multiple MLB models incrementally with new data.

    Handles the batch update of all MLB models using continuation training.
    For new MLBs not in existing models, trains from scratch.

    Args:
        existing_models: Dictionary of existing trained models keyed by MLB
        new_data: DataFrame containing new data for all MLBs
        additional_rounds: Number of additional boosting rounds for continuation

    Returns:
        Dictionary of updated models keyed by MLB
    """
    logger = logging.getLogger(__name__)
    updated_models = {}

    # Process each MLB in the new data
    for mlb in new_data["mlb"].unique():
        mlb_data = new_data[new_data["mlb"] == mlb].copy()

        # Ensure data is sorted by date
        if not isinstance(mlb_data.index, pd.DatetimeIndex):
            mlb_data.index = pd.to_datetime(mlb_data.index)
        mlb_data = mlb_data.sort_index()

        try:
            if mlb in existing_models:
                # Incremental update for existing MLB
                logger.info(f"Incrementally updating model for MLB {mlb}")
                updated_models[mlb] = update_model_incremental(
                    existing_models[mlb], mlb_data, additional_rounds=additional_rounds
                )
                logger.info(f"Successfully updated model for MLB {mlb}")

            else:
                # Train from scratch for new MLB
                logger.info(
                    f"Training new model for MLB {mlb} (not in existing models)"
                )
                updated_models[mlb] = _train_new_mlb_model(mlb_data)

        except Exception as e:
            logger.error(f"Failed to update model for MLB {mlb}: {e}")

            # Fallback: try to train from scratch
            try:
                logger.info(f"Attempting fallback training from scratch for MLB {mlb}")
                updated_models[mlb] = _train_new_mlb_model(mlb_data)
                logger.info(f"Fallback training successful for MLB {mlb}")
            except Exception as fallback_error:
                logger.error(
                    f"Fallback training also failed for MLB {mlb}: {fallback_error}"
                )
                # Keep original model if available, otherwise skip
                if mlb in existing_models:
                    logger.info(f"Keeping original model for MLB {mlb}")
                    updated_models[mlb] = existing_models[mlb]

    logger.info(f"Updated {len(updated_models)} MLB models")
    return updated_models


def validate_model_improvement(
    original_model: MultiOutputRegressor,
    updated_model: MultiOutputRegressor,
    validation_data: Tuple[pd.DataFrame, pd.DataFrame],
) -> Dict[str, float]:
    """
    Validate that incremental model update improved performance.

    Args:
        original_model: The original model before incremental update
        updated_model: The model after incremental update
        validation_data: Tuple of (X_val, y_val) for validation

    Returns:
        Dictionary with performance metrics and improvement indication
    """
    logger = logging.getLogger(__name__)

    try:
        X_val, y_val = validation_data

        # Get predictions from both models
        original_pred = original_model.predict(X_val)
        updated_pred = updated_model.predict(X_val)

        # Calculate RMSE for each output (forecast day)
        original_rmses = []
        updated_rmses = []

        for i in range(y_val.shape[1]):
            original_rmse = np.sqrt(
                mean_squared_error(y_val.iloc[:, i], original_pred[:, i])
            )
            updated_rmse = np.sqrt(
                mean_squared_error(y_val.iloc[:, i], updated_pred[:, i])
            )

            original_rmses.append(original_rmse)
            updated_rmses.append(updated_rmse)

        # Calculate average metrics
        avg_original_rmse = np.mean(original_rmses)
        avg_updated_rmse = np.mean(updated_rmses)
        improvement = avg_original_rmse - avg_updated_rmse
        improvement_pct = (improvement / avg_original_rmse) * 100

        results = {
            "original_rmse": avg_original_rmse,
            "updated_rmse": avg_updated_rmse,
            "improvement": improvement,
            "improvement_percentage": improvement_pct,
            "is_better": improvement > 0,
        }

        logger.info(
            f"Model validation - Original RMSE: {avg_original_rmse:.3f}, "
            f"Updated RMSE: {avg_updated_rmse:.3f}, "
            f"Improvement: {improvement_pct:.2f}%"
        )

        return results

    except Exception as e:
        logger.error(f"Model validation failed: {e}")
        return {
            "original_rmse": float("inf"),
            "updated_rmse": float("inf"),
            "improvement": 0,
            "improvement_percentage": 0,
            "is_better": False,
        }


def _train_new_mlb_model(
    mlb_data: pd.DataFrame, forecast_days: int = 90
) -> MultiOutputRegressor:
    """
    Train a new model for an MLB from scratch.

    This is a helper function used for new MLBs or fallback training.
    Uses the same training approach as forecast_future_sales_direct.

    Args:
        mlb_data: Data for a single MLB
        forecast_days: Number of days to forecast

    Returns:
        Trained MultiOutputRegressor model
    """
    logger = logging.getLogger(__name__)
    FEATURES = ["day_of_week", "day_of_month", "rolling_mean_3", "lag_1"]

    # Build training data using sliding window approach
    X_train = []
    y_train = []

    for i in range(len(mlb_data) - forecast_days):
        X_train.append(mlb_data.iloc[i][FEATURES].values)
        y_train.append(mlb_data.iloc[i + 1 : i + forecast_days + 1]["quant"].values)

    if len(X_train) == 0:
        raise ValueError("Insufficient data to train new model")

    X_train = pd.DataFrame(X_train, columns=FEATURES)
    y_columns = [f"quant_{i+1}" for i in range(forecast_days)]
    y_train = pd.DataFrame(y_train, columns=y_columns)

    # Train multi-output regressor
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

    logger.info(f"Trained new model with {len(X_train)} samples")
    return multi_model
