"""
Interactive forecasting script with user input for SKU selection.
Allows users to choose between importing fresh data or using cached data/forecasts.
"""

import logging
import pandas as pd
import plotly.express as px

from config import AppConfig
from database_utils import db_manager
from estimation.config_ml import BLD
from data_management.clean_sql_data import process_sales_data
from data_management.feature_creation import create_time_series_features
from estimation.model_forecast import save_regressors, forecast_future_sales_direct
from estimation.plot import print_available_skus

# Configure plotting backend
pd.options.plotting.backend = "matplotlib"

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_or_import_data(import_data: bool) -> pd.DataFrame:
    """Load data from file or import fresh data from database."""
    if import_data:
        logger.info("Importing fresh data from SQL...")
        data = db_manager.import_data_from_sql()

        # Save to CSV for future use
        data.to_csv(AppConfig.RAW_DATA_FILE, index=False)
        logger.info("Data imported and cached successfully!")
        return data
    else:
        logger.info("Loading cached data from CSV...")
        data = pd.read_csv(AppConfig.RAW_DATA_FILE, engine="pyarrow")
        logger.info("Cached data loaded successfully!")
        return data


def load_or_generate_forecasts(
    import_forecast: bool, feature_data: pd.DataFrame
) -> dict:
    """Load existing forecasts or generate new ones."""
    if import_forecast and AppConfig.SKU_FORECAST_FILE.exists():
        logger.info(f"Loading forecasts from {AppConfig.SKU_FORECAST_FILE}...")
        return pd.read_pickle(AppConfig.SKU_FORECAST_FILE)
    else:
        logger.info(f"Generating new {AppConfig.FORECAST_DAYS}-day forecasts...")
        sku_forecast = forecast_future_sales_direct(
            feature_data, AppConfig.FORECAST_DAYS
        )

        # Save forecasts for future use
        save_regressors(sku_forecast, AppConfig.SKU_FORECAST_FILE)
        logger.info("Forecasting complete and saved!")
        return sku_forecast


def interactive_sku_selection(sku_forecast: dict) -> None:
    """Handle interactive SKU selection and visualization."""
    print_available_skus(AppConfig.SKU_FORECAST_FILE)

    # Get user input for SKU
    sku = input("Please enter the SKU for predictions: ").strip()

    # Generate and save visualization
    forecast_df = sku_forecast.get(sku)

    if forecast_df is not None and not forecast_df.empty:
        fig = px.line(forecast_df, title=f"Forecast for SKU {sku}")
        html_file = BLD / f"{sku}_forecast.html"
        fig.write_html(html_file)
        logger.info(f"Forecast plotted and saved as HTML: {html_file}")
    else:
        logger.warning(f"No forecast available for SKU {sku}")


def main():
    """Main execution function for interactive forecasting."""
    try:
        # Configuration flags
        import_data = True
        import_forecast = True

        # Load or import data
        data = load_or_import_data(import_data)

        # Process data
        logger.info("Processing sales data...")
        clean_data = process_sales_data(data)

        logger.info("Creating time series features...")
        feature_data = create_time_series_features(clean_data)

        # Save feature data for debugging/analysis
        feature_data.to_csv(AppConfig.FEATURE_DATA_FILE, index=False)
        logger.info("Data processing complete!")

        # Load or generate forecasts
        sku_forecast = load_or_generate_forecasts(import_forecast, feature_data)

        # Interactive SKU selection and visualization
        interactive_sku_selection(sku_forecast)

        # Save forecasts to database
        logger.info("Saving forecasts to remote SQL database...")
        db_manager.save_forecasts_to_sql(sku_forecast)
        logger.info("Forecasts saved to database successfully!")

    except Exception as e:
        logger.error(f"Script failed with error: {e}")
        raise
    finally:
        db_manager.close_connection()


if __name__ == "__main__":
    main()
