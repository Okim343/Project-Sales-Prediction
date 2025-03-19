"""Code creating visual forecasting output."""

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

pd.options.mode.copy_on_write = True
pd.options.future.infer_string = True
pd.options.plotting.backend = "plotly"


def plot_predictions_from_model(pickle_path, test, df_actual, sku):
    """
    Create a Matplotlib figure showing actual data and predictions over time for a specific SKU.

    The cutoff date is determined as the first date in the test set (i.e., where the data was split
    into training and test).

    Parameters
    ----------
    pickle_path : Path or str
        Path to the pickled XGBoost regressor (e.g., 'reg.pkl').
    test : pd.DataFrame
        Test data with the features and a 'SKU' column.
    df_actual : pd.DataFrame
        DataFrame containing the actual data with a DateTime index, a 'quant' column, and a 'SKU' column.
    sku : str
        SKU identifier for filtering and plot title.

    Returns
    -------
    matplotlib.figure.Figure
        Matplotlib figure object with the plot.
    """
    pickle_path = Path(pickle_path)

    _fail_if_invalid_pickle_path(pickle_path)
    _fail_if_invalid_test_data(test)
    _fail_if_invalid_actual_data(df_actual)

    # Filter test and actual data for the given SKU
    test_sku = test[test["sku"] == sku].copy()
    df_actual_sku = df_actual[df_actual["sku"] == sku].copy()

    if test_sku.empty:
        raise ValueError(f"No test data found for SKU: {sku}")
    if df_actual_sku.empty:
        raise ValueError(f"No actual data found for SKU: {sku}")

    # Determine the cutoff date as the first date in the test set (i.e., the split date)
    cutoff_date = test_sku.index.min()

    reg = _load_regressor(pickle_path, sku)

    features = ["day_of_week", "day_of_month", "rolling_mean_3", "lag_1", "price"]
    x_test = test_sku[features]
    test_sku["prediction"] = reg.predict(x_test)

    df_merged = df_actual_sku.merge(
        test_sku[["prediction"]],
        how="left",
        left_index=True,
        right_index=True,
    )

    fig = _plot_actual_vs_predictions(df_merged, sku, cutoff_date)

    return fig


def _plot_actual_vs_predictions(df_merged, sku, cutoff_date):
    """
    Helper function to plot actual data and predictions using Matplotlib.

    Parameters
    ----------
    df_merged : pd.DataFrame
        DataFrame containing actual 'quant' and 'prediction' columns with DateTime index.
    sku : str
        SKU identifier for the plot title.
    cutoff_date : str or pd.Timestamp
        Date indicating the cutoff line on the plot.

    Returns
    -------
    matplotlib.figure.Figure
        Matplotlib figure object with the plot.
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(df_merged.index, df_merged["quant"], label="True Data", linewidth=2.5)
    ax.plot(
        df_merged.index, df_merged["prediction"], label="Predictions", linewidth=2.5
    )

    ax.axvline(
        pd.to_datetime(cutoff_date),
        color="black",
        linestyle="--",
        label="Cutoff Date",
        linewidth=2,
    )

    ax.set_title(f"Predicted Sales for SKU: {sku}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales Quantity")
    ax.legend()

    fig.tight_layout()

    return fig


def print_available_skus(pickle_path: Path) -> None:
    """
    Load the dictionary of regressors from a pickle file and print all the available SKUs.

    Args:
        pickle_path (Path): Path to the pickle file containing the dictionary of regressors.
    """
    with pickle_path.open("rb") as f:
        regressors = pickle.load(f)

    if not isinstance(regressors, dict):
        raise ValueError("The loaded object is not a dictionary.")

    print("Available SKUs:")
    for sku in regressors.keys():
        print(sku)


def _load_regressor(pickle_path, sku):
    with Path.open(pickle_path, "rb") as file:
        regressors = pickle.load(file)

    if sku not in regressors:
        raise ValueError(f"No regressor found for SKU: {sku}")

    regressor = regressors[sku]

    if not hasattr(regressor, "predict"):
        raise ValueError(
            f"Loaded object for SKU {sku} is not a valid XGBoost regressor."
        )

    return regressor


def _fail_if_invalid_pickle_path(pickle_path):
    """Raise an error if pickle_path is not a valid file path."""
    if not pickle_path.is_file():
        error_msg = (
            f"Invalid pickle path: {pickle_path} does not exist or is not a file."
        )
        raise ValueError(error_msg)


def _fail_if_invalid_test_data(test):
    """Raise an error if test data is not a DataFrame or missing required columns."""
    required_columns = {"day_of_week", "day_of_month", "rolling_mean_3", "lag_1", "sku"}
    if not isinstance(test, pd.DataFrame):
        error_msg = f"'test' must be a pandas DataFrame, got {type(test)}."
        raise TypeError(error_msg)

    if not required_columns.issubset(test.columns):
        missing_columns = required_columns - set(test.columns)
        error_msg = f"'test' DataFrame is missing required columns: {missing_columns}"
        raise ValueError(error_msg)


def _fail_if_invalid_actual_data(df_actual):
    """Raise an error if df_actual is not a DataFrame or missing required columns."""
    required_columns = {"quant", "sku"}
    if not isinstance(df_actual, pd.DataFrame):
        error_msg = f"'df_actual' must be a pandas DataFrame, got {type(df_actual)}."
        raise TypeError(error_msg)

    if not required_columns.issubset(df_actual.columns):
        missing_columns = required_columns - set(df_actual.columns)
        error_msg = (
            f"'df_actual' DataFrame is missing required columns: {missing_columns}"
        )
        raise ValueError(error_msg)
