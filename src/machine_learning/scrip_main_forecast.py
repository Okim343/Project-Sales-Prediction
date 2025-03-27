import logging
import pandas as pd
import plotly.express as px

from estimation.config_ml import DATA, BLD
from data_management.import_SQL import import_data_from_sql
from data_management.clean_sql_data import process_sales_data
from data_management.feature_creation import create_time_series_features
from estimation.model_forecast import forecast_future_sales_with_split, save_regressors
from estimation.plot import print_available_skus

pd.options.plotting.backend = "matplotlib"

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    import_data = False
    import_forecast = True
    if import_data:
        print("Importing data from SQL...")

        # Connection details
        user = "postgres"
        password = "Tommy627!"
        host = "172.27.40.210"
        port = "5432"
        dbname = "Mercado Livre"
        view = "public.view_enrico"

        data = import_data_from_sql(user, password, host, port, dbname, view)
        produces = DATA / "raw_sql.csv"
        data.to_csv(produces, index=False)

        print("Data imported successfully!")
    else:
        data = pd.read_csv(DATA / "raw_sql.csv", engine="pyarrow")
        print("Data already imported!")

    clean_data = process_sales_data(data)

    feature_data = create_time_series_features(clean_data)

    output_dir = BLD

    if (
        not output_dir.exists()
    ):  # creating a loop to check if the directory exists and create it if needed
        output_dir.mkdir()
        logging.info("Directory created")
    else:
        logging.info("BLD directory already exists")

    feature_data.to_csv(BLD / "feature_data.csv")

    # sku = "TC212"

    """fig = plot_timeseries_for_sku(feature_data, sku , "2024-01-01", "2025-01-01")
    produces = BLD / f"{sku}_time_series.html"
    fig.write_html(produces)
    print(f"Time series plotted and saved as HTML: {produces}") """

    logging.info("Data processing complete!")

    pickle_path = BLD / "sku_forecast.pkl"
    forecast_days = 30

    if import_forecast:
        logging.info(f"Loading regressors from {pickle_path}...")
        sku_forecast = pd.read_pickle(pickle_path)

    else:
        sku_forecast = forecast_future_sales_with_split(feature_data, forecast_days)

        # Define where to save the dictionary
        output_file = BLD / "sku_forecast.pkl"
        save_regressors(sku_forecast, output_file)

        logging.info("Model training and saving complete!")

    print_available_skus(pickle_path)

    # Prompt the user to enter the SKU
    sku = input("Please enter the SKU for predictions: ").strip()

    forecast_df = sku_forecast.get(sku)

    if forecast_df is not None:
        fig = px.line(forecast_df, title=f"Forecast for SKU {sku}")
        html_file = BLD / f"{sku}_forecast.html"
        fig.write_html(html_file)
        print(f"Forecast plotted and saved as HTML: {html_file}")
    else:
        print(f"No forecast available for SKU {sku}")
