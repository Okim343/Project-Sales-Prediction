"""Function(s) for cleaning the sales data set."""

import pandas as pd


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
    """Converts the 'date' column to datetime format and removes timezones."""
    data = data.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")  # Convert to datetime
    data["date"] = data["date"].apply(
        lambda x: x.tz_localize(None)
        if hasattr(x, "tzinfo") and x.tzinfo is not None
        else x
    )  # Remove timezone
    return data


def _collapse_sales_data(data: pd.DataFrame):
    """Aggregates sales data by date, summing the quantity."""
    return data.groupby("date", as_index=False).agg(
        {"quant": "sum", "sku": "first", "price": "sum"}
    )


def _set_datetime_index(data: pd.DataFrame):
    """Sets 'date' as index and converts it to a DatetimeIndex."""
    data = data.copy()
    data = data.set_index("date")
    data.index = pd.to_datetime(data.index, errors="coerce")
    return data


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
