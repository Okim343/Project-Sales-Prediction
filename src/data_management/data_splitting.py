"""Function(s) for splitting the data into training and testing."""

import pandas as pd

from final_project_enrico.data_management.clean_sales_data import _set_datetime_index


def split_train_test(data: pd.DataFrame, split_date: str):
    """Splits the dataset into training and testing sets based on a date threshold.

    Parameters:
    data (pd.DataFrame): The input DataFrame with a time-based index.
    split_date (str): The date threshold to split the data.

    Returns:
    The training and testing sets.
    """
    data = data.copy()
    data = _set_datetime_index(data)

    _fail_if_split_date_invalid(data, split_date)

    train = _get_train_data(data, split_date)
    test = _get_test_data(data, split_date)
    return train, test


def _get_train_data(data: pd.DataFrame, split_date: str):
    """Filters the training data (dates before the split date)."""
    split_date = pd.to_datetime(split_date)
    return data.loc[data.index < split_date].copy()


def _get_test_data(data: pd.DataFrame, split_date: str):
    """Filters the test data (dates on or after the split date)."""
    split_date = pd.to_datetime(split_date)
    return data.loc[data.index >= split_date].copy()


def _fail_if_split_date_invalid(data: pd.DataFrame, split_date: str):
    """Raise an error if split_date is not within the range of data's datetime index."""
    split_date = pd.to_datetime(split_date)

    if not isinstance(data.index, pd.DatetimeIndex):
        error_msg = "Data index must be a DatetimeIndex."
        raise TypeError(error_msg)

    min_date = data.index.min()
    max_date = data.index.max()

    if not (min_date <= split_date <= max_date):
        error_msg = f"split_date '{split_date}' is outside the available data range "
        raise ValueError(error_msg)
