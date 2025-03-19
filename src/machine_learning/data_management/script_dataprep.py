import logging
import pandas as pd

from config_ml import DATA, BLD
from import_SQL import import_data_from_sql
from clean_sql_data import process_sales_data
from feature_creation import create_time_series_features

logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":
    import_data = True
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

    print("Data processing complete!")
