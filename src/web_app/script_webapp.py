import dash
from dash import dcc, html, dash_table, Input, Output
import pandas as pd
import plotly.express as px
from sqlalchemy import create_engine

# Connect to your remote SQL Server
# Connection details
user = "postgres"
password = "Tommy627!"
host = "172.27.40.210"
port = "5432"
dbname = "Mercado Livre"
table = "public.sku_forecats_90_days"

engine = create_engine(f"postgresql://{user}:{password}@{host}:{port}/{dbname}")


# Read forecast data from the remote database
def get_forecast_data():
    query = f"SELECT * FROM {table}"
    df = pd.read_sql(query, engine)
    df["date"] = pd.to_datetime(df["date"])  # Ensure 'date' column is datetime
    return df


# Initial data load
df = get_forecast_data()

# Unique SKUs
sku_options = df["sku"].unique()

# Dash app setup
app = dash.Dash(__name__)
app.title = "SKU Forecast Dashboard"

app.layout = html.Div(
    [
        html.H1("Sales Forecast Dashboard"),
        html.Label("Choose SKU:"),
        dcc.Dropdown(
            id="sku-dropdown",
            options=[{"label": sku, "value": sku} for sku in sku_options],
            value=sku_options[0],
        ),
        html.Label("Select Date Range:"),
        dcc.DatePickerRange(
            id="date-picker",
            start_date=df["date"].min(),
            end_date=df["date"].max(),
            display_format="YYYY-MM-DD",
        ),
        dcc.Graph(id="forecast-graph"),
        html.H3("Forecast Table"),
        dash_table.DataTable(
            id="sku_forecast-table",
            columns=[{"name": col, "id": col} for col in ["date", "sku", "prediction"]],
            page_size=10,
            style_table={"overflowX": "auto"},
        ),
    ]
)


@app.callback(
    [Output("forecast-graph", "figure"), Output("sku_forecast-table", "data")],
    [
        Input("sku-dropdown", "value"),
        Input("date-picker", "start_date"),
        Input("date-picker", "end_date"),
    ],
)
def update_output(selected_sku, start_date, end_date):
    dff = get_forecast_data()
    dff = dff[
        (dff["sku"] == selected_sku)
        & (dff["date"] >= pd.to_datetime(start_date))
        & (dff["date"] <= pd.to_datetime(end_date))
    ]

    fig = px.line(dff, x="date", y="prediction", title=f"Forecast for {selected_sku}")
    return fig, dff[["date", "sku", "prediction"]].to_dict("records")


if __name__ == "__main__":
    app.run(debug=True)
