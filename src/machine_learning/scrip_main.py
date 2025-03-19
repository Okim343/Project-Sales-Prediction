import logging
import pandas as pd

from config_ml import DATA, BLD
from data_management.import_SQL import import_data_from_sql
from data_management.clean_sql_data import process_sales_data
from data_management.feature_creation import create_time_series_features
from estimation.model import train_model_for_each_sku, save_regressors
from estimation.plot import plot_predictions_from_model, print_available_skus
from estimation.data_splitting import split_train_test

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

    logging.info("Data processing complete!")

    sku_regressors = train_model_for_each_sku(feature_data)

    # Define where to save the dictionary
    output_file = BLD / "sku_regressors.pkl"
    save_regressors(sku_regressors, output_file)

    logging.info("Model training and saving complete!")

    pickle_path = BLD / "sku_regressors.pkl"

    print_available_skus(pickle_path)

    # Prompt the user to enter the SKU
    sku = input("Please enter the SKU for predictions: ").strip()

    produces = BLD / f"{sku}_predictions.png"

    sku_data = feature_data[feature_data["sku"] == sku].copy()

    if sku_data.empty:
        logging.error(f"No data found for SKU: {sku}.")
    else:
        train, test = split_train_test(sku_data)

        logging.info(f"Plotting predictions for SKU: {sku}...")

        fig = plot_predictions_from_model(pickle_path, test, feature_data, sku)
        fig.savefig(produces)

        logging.info("Predictions plotted and saved!")
