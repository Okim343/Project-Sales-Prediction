"""Plot a time series of the final data to visualize gaps in the data."""

import pandas as pd
import plotly.express as px


def plot_timeseries_for_sku(
    data: pd.DataFrame, sku: str, start_date: str, end_date: str
):
    """
    Plot the time series for a specific SKU between the given start_date and end_date using Plotly,
    showing gaps for missing data.

    Args:
        data (pd.DataFrame): The DataFrame containing sales information with columns including 'sku', 'quant',
                             and either a datetime index or a 'date' column.
        sku (str): The SKU identifier to filter the data.
        start_date (str): The start date of the period (in 'YYYY-MM-DD' format).
        end_date (str): The end date of the period (in 'YYYY-MM-DD' format).

    Returns:
        fig: The Plotly figure object containing the plot.
    """
    # Filter data for the specified SKU
    sku_data = data[data["sku"] == sku].copy()

    # Ensure the index is datetime; if not, use the 'date' column if available
    if not isinstance(sku_data.index, pd.DatetimeIndex):
        if "date" in sku_data.columns:
            sku_data["date"] = pd.to_datetime(sku_data["date"])
            sku_data.set_index("date", inplace=True)
        else:
            raise ValueError("Data must have a datetime index or a 'date' column.")

    # Filter data for the given time period
    mask = (sku_data.index >= start_date) & (sku_data.index <= end_date)
    sku_data = sku_data.loc[mask]

    # Reindex to include all dates in the period, leaving missing values as NaN
    full_date_range = pd.date_range(start=start_date, end=end_date, freq="D")
    sku_data = sku_data.reindex(full_date_range)
    sku_data.index.name = "date"
    # Fill the 'sku' column for missing rows
    sku_data["sku"] = sku

    if sku_data.empty:
        raise ValueError(
            f"No data found for SKU {sku} between {start_date} and {end_date}."
        )

    # Reset index for Plotly
    sku_data_reset = sku_data.reset_index()

    # Create a Plotly line chart; missing data (NaN) will create gaps if connectgaps is False
    fig = px.line(
        sku_data_reset,
        x="date",
        y="quant",
        title=f"Time Series for SKU {sku} from {start_date} to {end_date}",
        labels={"date": "Date", "quant": "Quantity"},
    )

    # Update traces to not connect gaps
    fig.update_traces(connectgaps=False)

    return fig
