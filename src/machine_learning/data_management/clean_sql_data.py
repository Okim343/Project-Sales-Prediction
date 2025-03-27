"""Function(s) for cleaning the sales data set."""

import pandas as pd
import numpy as np


def process_sales_data(data: pd.DataFrame):
    """Processes the sales data.

    This function renames columns, converts date formats, and aggregates quantities by date.

    Parameters:
    data (pd.DataFrame): The input dataframe with columns:
        "order_id",
        "date_created",
        "fulfilled",
        "order_items_item_seller_sku",
        "order_items_quantity",
        "order_items_unit_price",

    Returns:
    pd.DataFrame: Processed dataframe with cleaned and aggregated sales data.
    """
    data = data.copy()
    _fail_if_invalid_sales_data(data)

    data = _rename_columns(data)
    data = _convert_date_column(data)
    data = _collapse_sales_data(data)
    data = _set_datetime_index(data)
    data = _mark_missing_data(data)
    data = _remove_long_zero_periods(data)
    data = data.dropna(subset=["quant"])

    return data


def _rename_columns(data: pd.DataFrame):
    """Renames columns to more understandable names."""
    return data.rename(
        columns={
            "date_created": "date",
            "order_items_quantity": "quant",
            "order_items_unit_price": "price",
            "order_items_item_seller_sku": "sku",
        }
    ).copy()


def _convert_date_column(data: pd.DataFrame):
    """Converts the 'date' column to datetime format, removes timezones, and normalizes to midnight."""
    data = data.copy()
    # Convert the 'date' column to datetime
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    # Remove timezone: if the timestamp is timezone-aware, use tz_convert(None) to remove the tz info.
    data["date"] = data["date"].apply(
        lambda x: x.tz_convert(None) if x is not pd.NaT and x.tzinfo is not None else x
    )
    # Normalize the datetime to midnight (i.e., keep only the date part)
    data["date"] = data["date"].dt.normalize()
    return data


def _collapse_sales_data(data: pd.DataFrame):
    """Aggregates sales data by date and SKU, summing the quantity and price."""
    data = data.copy()
    # Ensure that the 'date' column is converted to datetime and normalized
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    # Group by both date and SKU
    return data.groupby(["date", "sku"], as_index=False).agg(
        {"quant": "sum", "price": "sum"}
    )


def _set_datetime_index(data: pd.DataFrame):
    """Sets 'date' as index and converts it to a DatetimeIndex."""
    data = data.copy()
    data = data.set_index("date")
    data.index = pd.to_datetime(data.index, errors="coerce")
    return data


def _mark_missing_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    For each SKU in the data, reindex the DataFrame so that every day between the first and
    last available date is present. For missing days, the "quant" and "price" values will be
    set to 0, while the "sku" column will be filled appropriately.

    Args:
        data (pd.DataFrame): DataFrame with a datetime index and a "sku" column.

    Returns:
        pd.DataFrame: Reindexed DataFrame with all days present for each SKU.
    """
    df_list = []
    for sku in data["sku"].unique():
        sku_data = data[data["sku"] == sku].sort_index()
        # Drop rows with invalid (NaT) dates in the index
        sku_data = sku_data[sku_data.index.notna()]
        if sku_data.empty:
            continue

        # Normalize the index so that timestamps become midnight
        sku_data.index = sku_data.index.normalize()
        # Remove duplicate date entries (keeping the first occurrence)
        sku_data = sku_data[~sku_data.index.duplicated(keep="first")]

        start_date = sku_data.index.min()
        end_date = sku_data.index.max()
        if pd.isna(start_date) or pd.isna(end_date):
            continue

        full_date_range = pd.date_range(start=start_date, end=end_date, freq="D")
        sku_data_reindexed = sku_data.reindex(full_date_range)
        # Fill the "sku" column for missing days with the current SKU value.
        sku_data_reindexed["sku"] = sku
        # Fill missing 'quant' and 'price' with 0
        sku_data_reindexed["quant"] = sku_data_reindexed["quant"].fillna(0)
        sku_data_reindexed["price"] = sku_data_reindexed["price"].fillna(0)
        sku_data_reindexed.index.name = "date"
        df_list.append(sku_data_reindexed)

    if df_list:
        return pd.concat(df_list)
    else:
        return data.copy()


def _remove_long_zero_periods(
    data: pd.DataFrame, zero_period_days: int = 7
) -> pd.DataFrame:
    """
    For each SKU in the data, for periods where 'quant' is 0 continuously for longer than
    zero_period_days, set the 'quant' value to NaN. This indicates that the sales data in that
    period is missing or unreliable, so the model can handle it appropriately.

    Args:
        data (pd.DataFrame): DataFrame with a datetime index and a "sku" column.
        zero_period_days (int): Minimum number of consecutive days with quant equal to 0 to trigger setting to NaN.

    Returns:
        pd.DataFrame: DataFrame with long zero periods marked as missing (NaN).
    """
    df_list = []
    for sku in data["sku"].unique():
        sku_data = data[data["sku"] == sku].sort_index()
        # Create a mask for rows where quant is 0
        zero_mask = sku_data["quant"] == 0
        zero_data = sku_data[zero_mask]
        if zero_data.empty:
            df_list.append(sku_data)
            continue

        # Group consecutive dates in zero_data: if the difference between consecutive dates
        # is not exactly one day, then it's a new group.
        groups = (zero_data.index.to_series().diff() != pd.Timedelta(days=1)).cumsum()

        # For each group of consecutive zeros, if the group is longer than the threshold, mark those rows as missing
        for _, group in zero_data.groupby(groups):
            if len(group) > zero_period_days:
                sku_data.loc[group.index, "quant"] = np.nan
        df_list.append(sku_data)
    if df_list:
        return pd.concat(df_list).sort_index()
    else:
        return data.copy()


def _fail_if_invalid_sales_data(data: pd.DataFrame):
    """Raise an error if data is not a DataFrame or is missing required columns."""
    required_columns = {
        "order_id",
        "date_created",
        "fulfilled",
        "order_items_item_seller_sku",
        "order_items_quantity",
        "order_items_unit_price",
    }

    if not isinstance(data, pd.DataFrame):
        error_msg = f"'data' must be a pandas DataFrame, got {type(data)}."
        raise TypeError(error_msg)

    if not required_columns.issubset(data.columns):
        missing_columns = required_columns - set(data.columns)
        error_msg = f"'data' DataFrame is missing required columns: {missing_columns}"
        raise ValueError(error_msg)
