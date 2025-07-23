import logging
import pandas as pd

from data_management.import_SQL import import_data_from_sql
from data_management.clean_sql_data import process_sales_data
from data_management.feature_creation import create_time_series_features
from estimation.model_forecast import forecast_future_sales_direct
from post_estimation.save_sql import save_forecasts_to_sql

pd.options.plotting.backend = "matplotlib"

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    print("Importing data from SQL...")

    # Connection details
    user = "postgres"
    password = "Tommy627!"
    host = "172.27.40.210"
    port = "5432"
    dbname = "Mercado Livre"
    view = "public.view_enrico"

    data = import_data_from_sql(user, password, host, port, dbname, view)

    print("Data imported successfully!")

    clean_data = process_sales_data(data)

    feature_data = create_time_series_features(clean_data)

    logging.info("Data processing complete!")

    forecast_days = 90

    sku_forecast = forecast_future_sales_direct(feature_data, forecast_days)

    logging.info("Model training and saving complete!")

    save_forecasts_to_sql(sku_forecast, "sku_forecats_90_days")
    logging.info("Forecasts saved to remote SQL database.")
