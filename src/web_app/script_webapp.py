"""
Sales Forecast Dashboard - Interactive web application for viewing SKU forecasts.
Features data caching, error handling, and optimized database queries.
"""

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import dash
from dash import dcc, html, dash_table, Input, Output
import pandas as pd
import plotly.express as px

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).parent.parent / "machine_learning"))

from config import DatabaseConfig
from database_utils import DatabaseManager

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global variables for caching
_cached_data: Optional[pd.DataFrame] = None
_cache_timestamp: Optional[datetime] = None
_cache_duration = timedelta(minutes=15)  # Cache for 15 minutes

# Database manager
db_manager = DatabaseManager()


def get_forecast_data(force_refresh: bool = False) -> pd.DataFrame:
    """
    Get forecast data with caching mechanism.

    Args:
        force_refresh: Force refresh the cache

    Returns:
        DataFrame with forecast data
    """
    global _cached_data, _cache_timestamp

    # Check if cache is valid
    now = datetime.now()
    cache_valid = (
        _cached_data is not None
        and _cache_timestamp is not None
        and (now - _cache_timestamp) < _cache_duration
        and not force_refresh
    )

    if cache_valid:
        logger.info("Using cached forecast data")
        return _cached_data

    try:
        logger.info("Fetching fresh forecast data from database...")
        query = f"SELECT * FROM {DatabaseConfig.FORECAST_TABLE}"
        df = pd.read_sql(query, db_manager.engine)
        df["date"] = pd.to_datetime(df["date"])

        # Update cache
        _cached_data = df
        _cache_timestamp = now

        logger.info(f"Loaded {len(df)} forecast records")
        return df

    except Exception as e:
        logger.error(f"Failed to fetch forecast data: {e}")
        # Return cached data if available, otherwise empty DataFrame
        if _cached_data is not None:
            logger.warning("Using stale cached data due to database error")
            return _cached_data
        else:
            logger.error("No cached data available, returning empty DataFrame")
            return pd.DataFrame(columns=["date", "sku", "prediction"])


def get_unique_skus() -> List[str]:
    """Get list of unique SKUs from forecast data."""
    try:
        df = get_forecast_data()
        return sorted(df["sku"].unique()) if not df.empty else []
    except Exception as e:
        logger.error(f"Failed to get unique SKUs: {e}")
        return []


def get_sku_date_range(sku: str) -> tuple[datetime.date, datetime.date]:
    """
    Get the date range (min, max) for a specific SKU.

    Args:
        sku: The SKU to get date range for

    Returns:
        Tuple of (min_date, max_date) for the SKU, or global range if SKU not found
    """
    try:
        df = get_forecast_data()
        if df.empty:
            # Return current date if no data
            current_date = datetime.now().date()
            return current_date, current_date

        if sku:
            # Filter data for specific SKU
            sku_data = df[df["sku"] == sku]
            if not sku_data.empty:
                min_date = sku_data["date"].min().date()
                max_date = sku_data["date"].max().date()
                return min_date, max_date

        # Fallback to global range if SKU not found or not specified
        min_date = df["date"].min().date()
        max_date = df["date"].max().date()
        return min_date, max_date

    except Exception as e:
        logger.error(f"Failed to get date range for SKU {sku}: {e}")
        current_date = datetime.now().date()
        return current_date, current_date


# Initialize the Dash app
def create_app() -> dash.Dash:
    """Create and configure the Dash application."""
    app = dash.Dash(__name__)
    app.title = "SKU Forecast Dashboard"

    # Get initial data for layout
    try:
        df = get_forecast_data()
        sku_options = get_unique_skus()

        if df.empty or not sku_options:
            logger.warning("No forecast data available for dashboard initialization")
            # Create minimal layout with no data message
            app.layout = create_no_data_layout()
        else:
            app.layout = create_main_layout(df, sku_options)

    except Exception as e:
        logger.error(f"Failed to initialize app layout: {e}")
        app.layout = create_error_layout(str(e))

    return app


def create_main_layout(df: pd.DataFrame, sku_options: List[str]) -> html.Div:
    """Create the main dashboard layout."""
    return html.Div(
        [
            html.Div(
                [
                    html.H1("Sales Forecast Dashboard", className="dashboard-title"),
                    html.P(
                        f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                        className="last-updated",
                    ),
                ],
                className="header",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Choose SKU:", className="control-label"),
                            dcc.Dropdown(
                                id="sku-dropdown",
                                options=[
                                    {"label": sku, "value": sku} for sku in sku_options
                                ],
                                value=sku_options[0] if sku_options else None,
                                className="control-input",
                            ),
                        ],
                        className="control-group",
                    ),
                    html.Div(
                        [
                            html.Label("Select Date Range:", className="control-label"),
                            dcc.DatePickerRange(
                                id="date-picker",
                                start_date=df["date"].min()
                                if not df.empty
                                else datetime.now().date(),
                                end_date=df["date"].max()
                                if not df.empty
                                else datetime.now().date(),
                                min_date_allowed=df["date"].min()
                                if not df.empty
                                else datetime.now().date(),
                                max_date_allowed=df["date"].max()
                                if not df.empty
                                else datetime.now().date(),
                                display_format="YYYY-MM-DD",
                                className="control-input",
                            ),
                        ],
                        className="control-group",
                    ),
                    html.Button(
                        "Refresh Data", id="refresh-button", className="refresh-btn"
                    ),
                ],
                className="controls",
            ),
            dcc.Graph(id="forecast-graph", className="main-graph"),
            html.Div(
                [
                    html.H3("Forecast Data Table"),
                    dash_table.DataTable(
                        id="forecast-table",
                        columns=[
                            {"name": "Date", "id": "date", "type": "datetime"},
                            {"name": "SKU", "id": "sku"},
                            {
                                "name": "Prediction",
                                "id": "prediction",
                                "type": "numeric",
                                "format": {"specifier": ".2f"},
                            },
                        ],
                        page_size=15,
                        sort_action="native",
                        filter_action="native",
                        style_table={"overflowX": "auto"},
                        style_cell={"textAlign": "left"},
                        style_data_conditional=[
                            {
                                "if": {"row_index": "odd"},
                                "backgroundColor": "rgb(248, 248, 248)",
                            }
                        ],
                    ),
                ],
                className="table-section",
            ),
            dcc.Store(id="data-store"),  # For storing cached data
        ],
        className="container",
    )


def create_no_data_layout() -> html.Div:
    """Create layout when no data is available."""
    return html.Div(
        [
            html.H1("Sales Forecast Dashboard"),
            html.Div(
                [
                    html.H3("No Forecast Data Available"),
                    html.P(
                        "Please check your data source and try refreshing the page."
                    ),
                    html.Button(
                        "Refresh", id="refresh-button", className="refresh-btn"
                    ),
                ],
                className="no-data-message",
            ),
        ]
    )


def create_error_layout(error_message: str) -> html.Div:
    """Create layout when there's an error."""
    return html.Div(
        [
            html.H1("Sales Forecast Dashboard"),
            html.Div(
                [
                    html.H3("Error Loading Dashboard"),
                    html.P(f"Error: {error_message}"),
                    html.P("Please check your configuration and try again."),
                ],
                className="error-message",
            ),
        ]
    )


# Create the app instance
app = create_app()


@app.callback(
    [
        Output("date-picker", "min_date_allowed"),
        Output("date-picker", "max_date_allowed"),
        Output("date-picker", "start_date"),
        Output("date-picker", "end_date"),
    ],
    [Input("sku-dropdown", "value")],
    prevent_initial_call=False,
)
def update_date_picker(selected_sku: str):
    """Update date picker range based on selected SKU."""
    try:
        if not selected_sku:
            # If no SKU selected, use global range
            df = get_forecast_data()
            if df.empty:
                current_date = datetime.now().date()
                return current_date, current_date, current_date, current_date

            min_date = df["date"].min().date()
            max_date = df["date"].max().date()
            return min_date, max_date, min_date, max_date

        # Get SKU-specific date range
        min_date, max_date = get_sku_date_range(selected_sku)

        # Return the date constraints and default start/end dates
        return min_date, max_date, min_date, max_date

    except Exception as e:
        logger.error(f"Error updating date picker for SKU {selected_sku}: {e}")
        current_date = datetime.now().date()
        return current_date, current_date, current_date, current_date


@app.callback(
    [Output("forecast-graph", "figure"), Output("forecast-table", "data")],
    [
        Input("sku-dropdown", "value"),
        Input("date-picker", "start_date"),
        Input("date-picker", "end_date"),
        Input("refresh-button", "n_clicks"),
    ],
    prevent_initial_call=False,
)
def update_dashboard(selected_sku: str, start_date: str, end_date: str, n_clicks: int):
    """Update the dashboard based on user inputs."""
    try:
        # Force refresh if refresh button was clicked
        force_refresh = n_clicks is not None and n_clicks > 0

        # Get forecast data (with caching)
        df = get_forecast_data(force_refresh=force_refresh)

        if df.empty:
            # Return empty plot if no data
            empty_fig = px.line(title="No Data Available")
            return empty_fig, []

        # Filter data based on selections
        filtered_df = df.copy()

        if selected_sku:
            filtered_df = filtered_df[filtered_df["sku"] == selected_sku]

        if start_date and end_date:
            filtered_df = filtered_df[
                (filtered_df["date"] >= pd.to_datetime(start_date))
                & (filtered_df["date"] <= pd.to_datetime(end_date))
            ]

        # Create the forecast plot
        if not filtered_df.empty:
            fig = px.line(
                filtered_df,
                x="date",
                y="prediction",
                title=f"Sales Forecast for SKU: {selected_sku}"
                if selected_sku
                else "Sales Forecast",
                labels={"prediction": "Predicted Sales", "date": "Date"},
            )
            fig.update_layout(
                xaxis_title="Date", yaxis_title="Predicted Sales", hovermode="x unified"
            )
        else:
            fig = px.line(title=f"No data available for SKU: {selected_sku}")

        # Prepare table data
        table_data = filtered_df[["date", "sku", "prediction"]].to_dict("records")

        return fig, table_data

    except Exception as e:
        logger.error(f"Error updating dashboard: {e}")
        error_fig = px.line(title=f"Error: {str(e)}")
        return error_fig, []


def main():
    """Main function to run the web application."""
    try:
        logger.info("Starting Sales Forecast Dashboard...")

        # Test database connection
        if not db_manager.test_connection():
            logger.error("Database connection failed. Check your configuration.")
            return

        logger.info("Database connection successful")
        logger.info("Dashboard available at: http://127.0.0.1:8050")

        # Run the app
        app.run(
            debug=False,  # Set to True for development
            host="127.0.0.1",
            port=8050,
            dev_tools_hot_reload=False,
        )

    except Exception as e:
        logger.error(f"Failed to start dashboard: {e}")
        raise
    finally:
        db_manager.close_connection()


if __name__ == "__main__":
    main()
