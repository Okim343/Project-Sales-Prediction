import pandas as pd
import logging

from config_ml import BLD

from model import train_model_for_each_sku, save_regressors
from plot import plot_predictions_from_model
from data_splitting import split_train_test

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    data = pd.read_csv(BLD / "feature_data.csv", parse_dates=["date"], index_col="date")

    sku_regressors = train_model_for_each_sku(data)

    # Define where to save the dictionary
    output_file = BLD / "sku_regressors.pkl"
    save_regressors(sku_regressors, output_file)

    print("Model training and saving complete!")

    sku = "AC033"
    pickle_path = BLD / "sku_regressors.pkl"
    df_actual = pd.read_csv(
        BLD / "feature_data.csv",
        engine="pyarrow",
        parse_dates=["date"],
        index_col="date",
    )
    produces = BLD / f"{sku}_predictions.png"

    # print_available_skus(pickle_path)

    sku_data = df_actual[df_actual["sku"] == sku].copy()

    train, test = split_train_test(sku_data)

    fig = plot_predictions_from_model(pickle_path, test, df_actual, sku)

    fig.savefig(produces)

    print("predictions plotted and saved!")
