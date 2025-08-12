"""
TEST Sales Forecast Dashboard - Interactive web application for viewing TEST SKU forecasts.
Features data caching, error handling, and optimized database queries.
USES TEST DATA FROM THE TEST FORECAST TABLE FOR DEVELOPMENT/TESTING PURPOSES.
"""

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import dash
from dash import dcc, html, dash_table, Input, Output
import dash.dependencies
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
    Get TEST forecast data with caching mechanism.

    Args:
        force_refresh: Force refresh the cache

    Returns:
        DataFrame with forecast data from TEST table
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
        logger.info("Using cached TEST forecast data")
        return _cached_data  # type: ignore

    try:
        logger.info("Fetching fresh TEST forecast data from database...")
        query = f"SELECT * FROM {DatabaseConfig.TEST_FORECAST_TABLE}"
        df = pd.read_sql(query, db_manager.engine)
        df["date"] = pd.to_datetime(df["date"])

        # Update cache
        _cached_data = df
        _cache_timestamp = now

        logger.info(f"Loaded {len(df)} TEST forecast records")
        return df

    except Exception as e:
        logger.error(f"Failed to fetch TEST forecast data: {e}")
        # Return cached data if available, otherwise empty DataFrame
        if _cached_data is not None:
            logger.warning("Using stale cached TEST data due to database error")
            return _cached_data
        else:
            logger.error("No cached TEST data available, returning empty DataFrame")
            return pd.DataFrame(columns=["date", "mlb", "sku", "prediction"])


def get_unique_mlbs() -> List[str]:
    """Get list of unique MLBs from TEST forecast data."""
    try:
        df = get_forecast_data()
        return sorted(df["mlb"].unique()) if not df.empty else []
    except Exception as e:
        logger.error(f"Failed to get unique MLBs from TEST data: {e}")
        return []


def get_mlb_date_range(mlb: str) -> tuple[datetime.date, datetime.date]:
    """
    Get the date range (min, max) for a specific MLB from TEST data.

    Args:
        mlb: The MLB to get date range for

    Returns:
        Tuple of (min_date, max_date) for the MLB, or global range if MLB not found
    """
    try:
        df = get_forecast_data()
        if df.empty:
            # Return current date if no data
            current_date = datetime.now().date()
            return current_date, current_date

        if mlb:
            # Filter data for specific MLB
            mlb_data = df[df["mlb"] == mlb]
            if not mlb_data.empty:
                min_date = mlb_data["date"].min().date()
                max_date = mlb_data["date"].max().date()
                return min_date, max_date

        # Fallback to global range if MLB not found or not specified
        min_date = df["date"].min().date()
        max_date = df["date"].max().date()
        return min_date, max_date

    except Exception as e:
        logger.error(f"Failed to get date range for MLB {mlb} from TEST data: {e}")
        current_date = datetime.now().date()
        return current_date, current_date


# Initialize the Dash app
def create_app() -> dash.Dash:
    """Create and configure the TEST Dash application."""
    app = dash.Dash(__name__)
    app.title = "TEST MLB Forecast Dashboard"

    # Create layout with authentication wrapper
    app.layout = create_auth_wrapper()

    return app


def create_auth_wrapper() -> html.Div:
    """Create the authentication wrapper layout."""
    return html.Div(
        [
            dcc.Store(id="session-store", storage_type="session"),
            html.Div(id="page-content"),
        ]
    )


def create_login_layout() -> html.Div:
    """Create the login layout for TEST version."""
    return html.Div(
        [
            html.Div(
                [
                    html.H1(
                        "TEST Sales Forecast Dashboard", className="dashboard-title"
                    ),
                    html.P(
                        "⚠️ This is the TEST version using test data ⚠️",
                        style={"color": "orange", "font-weight": "bold"},
                    ),
                    html.Div(
                        [
                            html.H3("Please enter password to access:"),
                            dcc.Input(
                                id="password-input",
                                type="password",
                                placeholder="Enter password...",
                                style={
                                    "padding": "10px",
                                    "margin": "10px",
                                    "width": "200px",
                                },
                            ),
                            html.Button(
                                "Login",
                                id="login-button",
                                style={"padding": "10px 20px", "margin": "10px"},
                            ),
                            html.Div(
                                id="login-status",
                                style={"color": "red", "margin": "10px"},
                            ),
                        ],
                        style={"text-align": "center", "padding": "50px"},
                    ),
                ],
                style={"text-align": "center", "margin-top": "100px"},
            )
        ]
    )


def create_main_layout(df: pd.DataFrame, mlb_options: List[str]) -> html.Div:
    """Create the main TEST dashboard layout."""
    return html.Div(
        [
            html.Div(
                [
                    html.H1(
                        "TEST Sales Forecast Dashboard", className="dashboard-title"
                    ),
                    html.P(
                        "⚠️ This is the TEST version using test data ⚠️",
                        style={
                            "color": "orange",
                            "font-weight": "bold",
                            "font-size": "16px",
                        },
                    ),
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
                            html.Label("Choose MLB:", className="control-label"),
                            dcc.Dropdown(
                                id="mlb-dropdown",
                                options=[
                                    {"label": mlb, "value": mlb} for mlb in mlb_options
                                ],
                                value=mlb_options[0] if mlb_options else None,
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
                    html.Button(
                        "Logout",
                        id="logout-button",
                        className="refresh-btn",
                        style={"margin-left": "10px", "background-color": "#dc3545"},
                    ),
                ],
                className="controls",
            ),
            dcc.Graph(id="forecast-graph", className="main-graph"),
            html.Div(
                [
                    html.H3("TEST Forecast Data Table"),
                    dash_table.DataTable(
                        id="forecast-table",
                        columns=[
                            {"name": "Date", "id": "date", "type": "datetime"},
                            {"name": "MLB", "id": "mlb"},
                            {"name": "SKU", "id": "sku"},
                            {
                                "name": "Prediction",
                                "id": "prediction",
                                "type": "numeric",
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
    """Create layout when no TEST data is available."""
    return html.Div(
        [
            html.H1("TEST Sales Forecast Dashboard"),
            html.P(
                "⚠️ This is the TEST version using test data ⚠️",
                style={"color": "orange", "font-weight": "bold"},
            ),
            html.Div(
                [
                    html.H3("No TEST Forecast Data Available"),
                    html.P(
                        "Please check your TEST data source and try refreshing the page."
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
    """Create layout when there's an error with TEST data."""
    return html.Div(
        [
            html.H1("TEST Sales Forecast Dashboard"),
            html.P(
                "⚠️ This is the TEST version using test data ⚠️",
                style={"color": "orange", "font-weight": "bold"},
            ),
            html.Div(
                [
                    html.H3("Error Loading TEST Dashboard"),
                    html.P(f"Error: {error_message}"),
                    html.P("Please check your TEST configuration and try again."),
                ],
                className="error-message",
            ),
        ]
    )


# Create the app instance
app = create_app()


# Authentication callbacks
@app.callback(
    [Output("session-store", "data"), Output("login-status", "children")],
    [Input("login-button", "n_clicks")],
    [dash.dependencies.State("password-input", "value")],
)
def authenticate_user(n_clicks, password):
    """Authenticate user with password."""
    if n_clicks is None:
        return {"authenticated": False}, ""

    if password == "thienri":
        return {"authenticated": True}, ""
    else:
        return {"authenticated": False}, "Incorrect password. Please try again."


@app.callback(
    Output("session-store", "data", allow_duplicate=True),
    [Input("logout-button", "n_clicks")],
    prevent_initial_call=True,
)
def logout_user(n_clicks):
    """Logout user by clearing session."""
    if n_clicks:
        return {"authenticated": False}
    return dash.no_update


@app.callback(Output("page-content", "children"), [Input("session-store", "data")])
def display_page(session_data):
    """Display appropriate page based on authentication status."""
    if session_data is None:
        session_data = {"authenticated": False}

    if session_data.get("authenticated", False):
        # User is authenticated, show main dashboard
        try:
            df = get_forecast_data()
            mlb_options = get_unique_mlbs()

            if df.empty or not mlb_options:
                logger.warning(
                    "No TEST forecast data available for dashboard initialization"
                )
                return create_no_data_layout()
            else:
                return create_main_layout(df, mlb_options)

        except Exception as e:
            logger.error(f"Failed to load TEST dashboard: {e}")
            return create_error_layout(str(e))
    else:
        # User is not authenticated, show login page
        return create_login_layout()


@app.callback(
    [
        Output("date-picker", "min_date_allowed"),
        Output("date-picker", "max_date_allowed"),
        Output("date-picker", "start_date"),
        Output("date-picker", "end_date"),
    ],
    [Input("mlb-dropdown", "value")],
    prevent_initial_call=False,
)
def update_date_picker(selected_mlb: str):
    """Update date picker range based on selected SKU from TEST data."""
    try:
        if not selected_mlb:
            # If no MLB selected, use global range
            df = get_forecast_data()
            if df.empty:
                current_date = datetime.now().date()
                return current_date, current_date, current_date, current_date

            min_date = df["date"].min().date()
            max_date = df["date"].max().date()
            return min_date, max_date, min_date, max_date

        # Get MLB-specific date range
        min_date, max_date = get_mlb_date_range(selected_mlb)

        # Return the date constraints and default start/end dates
        return min_date, max_date, min_date, max_date

    except Exception as e:
        logger.error(
            f"Error updating date picker for MLB {selected_mlb} from TEST data: {e}"
        )
        current_date = datetime.now().date()
        return current_date, current_date, current_date, current_date


@app.callback(
    [Output("forecast-graph", "figure"), Output("forecast-table", "data")],
    [
        Input("mlb-dropdown", "value"),
        Input("date-picker", "start_date"),
        Input("date-picker", "end_date"),
        Input("refresh-button", "n_clicks"),
    ],
    prevent_initial_call=False,
)
def update_dashboard(selected_mlb: str, start_date: str, end_date: str, n_clicks: int):
    """Update the TEST dashboard based on user inputs."""
    try:
        # Force refresh if refresh button was clicked
        force_refresh = n_clicks is not None and n_clicks > 0

        # Get TEST forecast data (with caching)
        df = get_forecast_data(force_refresh=force_refresh)

        if df.empty:
            # Return empty plot if no data
            empty_fig = px.line(title="No TEST Data Available")
            return empty_fig, []

        # Filter data based on selections
        filtered_df = df.copy()

        if selected_mlb:
            filtered_df = filtered_df[filtered_df["mlb"] == selected_mlb]

        if start_date and end_date:
            filtered_df = filtered_df[
                (filtered_df["date"] >= pd.to_datetime(start_date))
                & (filtered_df["date"] <= pd.to_datetime(end_date))
            ]

        # Create the forecast plot
        if not filtered_df.empty:
            # Get SKU for display purposes
            sku_for_display = (
                filtered_df["sku"].iloc[0]
                if "sku" in filtered_df.columns
                else "Unknown"
            )

            fig = px.line(
                filtered_df,
                x="date",
                y="prediction",
                title=f"TEST Sales Forecast for MLB: {selected_mlb} (SKU: {sku_for_display})"
                if selected_mlb
                else "TEST Sales Forecast",
                labels={"prediction": "Predicted Sales", "date": "Date"},
            )
            fig.update_layout(
                xaxis_title="Date", yaxis_title="Predicted Sales", hovermode="x unified"
            )
        else:
            fig = px.line(title=f"No TEST data available for MLB: {selected_mlb}")

        # Prepare table data
        table_data = filtered_df[["date", "mlb", "sku", "prediction"]].to_dict(
            "records"
        )

        return fig, table_data

    except Exception as e:
        logger.error(f"Error updating TEST dashboard: {e}")
        error_fig = px.line(title=f"TEST Dashboard Error: {str(e)}")
        return error_fig, []


def main():
    """Main function to run the TEST web application."""
    try:
        logger.info("Starting TEST Sales Forecast Dashboard...")

        # Test database connection
        if not db_manager.test_connection():
            logger.error("Database connection failed. Check your configuration.")
            return

        logger.info("Database connection successful")
        logger.info("TEST Dashboard available at: http://0.0.0.0:8051")
        logger.info("External access available at: http://YOUR_IP_ADDRESS:8051")

        # Run the app on different port (8051) to avoid conflict with production
        app.run(
            debug=False,  # Set to True for development
            host="0.0.0.0",
            port=8051,  # Different port from production
            dev_tools_hot_reload=False,
        )

    except Exception as e:
        logger.error(f"Failed to start TEST dashboard: {e}")
        raise
    finally:
        db_manager.close_connection()


if __name__ == "__main__":
    main()
