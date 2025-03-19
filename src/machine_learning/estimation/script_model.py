import pandas as pd

from config_ml import BLD

from model import train_model_for_each_sku, save_regressors


if __name__ == "__main__":
    data = pd.read_csv(BLD / "feature_data.csv")

    sku_regressors = train_model_for_each_sku(data)

    # Define where to save the dictionary
    output_file = BLD / "sku_regressors.pkl"
    save_regressors(sku_regressors, output_file)

    print("Model training and saving complete!")
