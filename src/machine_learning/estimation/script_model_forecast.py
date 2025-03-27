import pandas as pd
import logging
from config_ml import BLD
from model_forecast import forecast_future_sales_with_split, save_regressors
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    data = pd.read_csv(BLD / "feature_data.csv", parse_dates=["date"], index_col="date")

    # Define forecast period (e.g., 30 days forecast)
    forecast_days = 30

    # Generate forecasts for each SKU using the train/test split approach
    sku_forecast = forecast_future_sales_with_split(data, forecast_days)

    # Save the forecasts dictionary to a pickle file
    output_file = BLD / "sku_forecasts.pkl"
    save_regressors(sku_forecast, output_file)

    print("Model training and forecasting complete!")

    # Optionally, plot forecast for a specific SKU
    sku = "TC213"
    forecast_df = sku_forecast.get(sku)
    produces = BLD / f"{sku}_forecast.png"

    if forecast_df is not None:
        fig, ax = plt.subplots()
        forecast_df.plot(ax=ax)
        ax.set_title(f"Forecast for SKU {sku}")
        fig.savefig(produces)
        print("Forecast plotted and saved!")
    else:
        print(f"No forecast available for SKU {sku}")
