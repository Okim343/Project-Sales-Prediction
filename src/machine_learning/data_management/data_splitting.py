"""Function(s) for splitting the data into training and testing."""

import pandas as pd


def split_train_test(data: pd.DataFrame):
    """
    Splits the dataset into training and testing sets using the last month of data as the test set.

    The last month of data is determined by subtracting one month from the maximum date in the data index.
    The training set contains data with dates before the threshold, and the test set includes data on or after the threshold.

    Parameters:
        data (pd.DataFrame): The input DataFrame with a DatetimeIndex.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: The training and testing sets.
    """
    data = data.copy()

    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError("Data index must be a DatetimeIndex.")

    # Determine the last date in the dataset
    max_date = data.index.max()

    # Calculate the threshold as one month prior to the max_date
    threshold_date = max_date - pd.DateOffset(months=1)

    # Split the data into training and test sets
    train = data.loc[data.index < threshold_date].copy()
    test = data.loc[data.index >= threshold_date].copy()

    return train, test
