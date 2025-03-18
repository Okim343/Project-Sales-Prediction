"""Function(s) for creating features for machine learning."""

import pandas as pd


def create_time_series_features(data: pd.DataFrame):
    """Generates time series features based on the DataFrame index.

    Parameters:
    data (pd.DataFrame): The input DataFrame with a time-based index and 'quant' column.

    Returns:
    data: A new DataFrame with additional time series features.
    """
    data = data.copy()
    _fail_if_invalid_sales_data(data)

    data = _add_lag_features(data)
    data = _add_rolling_features(data)
    data = _add_calendar_features(data)
    return data


def _add_lag_features(data: pd.DataFrame):
    """Adds lag features to the dataset."""
    data["lag_1"] = data["quant"].shift(1)
    return data


def _add_rolling_features(data: pd.DataFrame):
    """Adds rolling window features to the dataset."""
    data["rolling_mean_3"] = data["quant"].rolling(window=3).mean()
    return data


def _add_calendar_features(data: pd.DataFrame):
    """Adds calendar-based time features from the DataFrame index."""
    data["day_of_week"] = data.index.dayofweek
    data["day_of_month"] = data.index.day
    return data


def _fail_if_invalid_sales_data(data: pd.DataFrame):
    """Raise an error if data is not a DataFrame."""
    if not isinstance(data, pd.DataFrame):
        error_msg = f"'data' must be a pandas DataFrame, got {type(data)}."
        raise TypeError(error_msg)
