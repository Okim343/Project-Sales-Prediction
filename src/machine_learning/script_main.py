"""
Main model training script with interactive SKU prediction visualization.
Handles data import, model training, and individual SKU predictions.
"""

import logging
import pandas as pd

from config import AppConfig
from database_utils import db_manager
from estimation.config_ml import BLD
from data_management.clean_sql_data import process_sales_data
from data_management.feature_creation import create_time_series_features
from estimation.model import train_model_for_each_sku, save_regressors
from estimation.plot import plot_predictions_from_model, print_available_skus
from estimation.data_splitting import split_train_test
from data_management.test_plot import plot_timeseries_for_sku

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def validate_data_freshness(data: pd.DataFrame, max_age_days: int = 7) -> None:
    """
    Validate that the cached data is fresh enough to use.
    Issues warnings if data is stale.

    Args:
        data: DataFrame containing the raw data
        max_age_days: Maximum acceptable age of data in days
    """
    try:
        # Determine date column
        date_col = None
        for col in ["date_created", "date"]:
            if col in data.columns:
                date_col = col
                break

        if not date_col:
            logger.warning(
                "No date column found in cached data - cannot validate freshness"
            )
            return

        # Parse dates and find the most recent
        dates = pd.to_datetime(data[date_col], errors="coerce")
        most_recent = dates.max()

        if pd.isna(most_recent):
            logger.warning("Could not parse dates in cached data")
            return

        # Calculate age of data
        today = pd.Timestamp.now().normalize()
        days_old = (today - most_recent.normalize()).days

        logger.info(
            f"Cached data age: {days_old} days (most recent: {most_recent.date()})"
        )

        if days_old > max_age_days:
            logger.warning(
                f"⚠️  STALE DATA WARNING: Cached data is {days_old} days old! "
                f"This may cause all SKUs to be marked as inactive. "
                f"Consider running with import_data=True or running refresh_data.py"
            )
        elif days_old > 2:
            logger.info(
                f"📅 Cached data is {days_old} days old - still usable but consider refreshing"
            )
        else:
            logger.info("✅ Cached data is fresh")

    except Exception as e:
        logger.warning(f"Error validating data freshness: {e}")


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

        # Check if cached file exists
        if not AppConfig.RAW_DATA_FILE.exists():
            logger.warning(f"Cached data file not found: {AppConfig.RAW_DATA_FILE}")
            logger.info("Falling back to importing fresh data from database...")
            data = db_manager.import_data_from_sql()
            data.to_csv(AppConfig.RAW_DATA_FILE, index=False)
            return data

        data = pd.read_csv(AppConfig.RAW_DATA_FILE, engine="pyarrow")

        # Validate data freshness
        validate_data_freshness(data)

        return data


def generate_sample_timeseries_plot(
    clean_data: pd.DataFrame, sku: str | None = None
) -> None:
    """Generate and save a sample time series plot for demonstration."""
    sku = sku if sku is not None else AppConfig.DEFAULT_SKU

    logger.info(f"Generating time series plot for SKU {sku}...")
    fig = plot_timeseries_for_sku(clean_data, sku, "2024-01-01", "2025-01-01")
    output_file = BLD / f"{sku}_time_series.html"
    fig.write_html(output_file)
    logger.info(f"Time series plot saved: {output_file}")


def load_or_train_models(import_regressor: bool, feature_data: pd.DataFrame) -> None:
    """Load existing models or train new ones."""
    if import_regressor and AppConfig.SKU_REGRESSORS_FILE.exists():
        logger.info(f"Models will be loaded from {AppConfig.SKU_REGRESSORS_FILE}")
    else:
        logger.info("Training new models for each SKU...")
        sku_regressors = train_model_for_each_sku(feature_data)

        # Save trained models
        save_regressors(sku_regressors, AppConfig.SKU_REGRESSORS_FILE)
        logger.info("Model training and saving complete!")


def interactive_prediction_visualization(feature_data: pd.DataFrame) -> None:
    """Handle interactive SKU selection and prediction visualization."""
    print_available_skus(AppConfig.SKU_REGRESSORS_FILE)

    # Get user input for SKU
    sku = input("Please enter the SKU for predictions: ").strip()

    # Check if SKU data exists
    sku_data = feature_data[feature_data["sku"] == sku].copy()

    if sku_data.empty:
        logger.error(f"No data found for SKU: {sku}")
        return

    # Generate predictions and visualization
    logger.info(f"Generating predictions for SKU: {sku}...")

    _, test = split_train_test(sku_data)

    fig = plot_predictions_from_model(
        AppConfig.SKU_REGRESSORS_FILE, test, feature_data, sku
    )

    output_file = BLD / f"{sku}_predictions.png"
    fig.savefig(output_file)
    logger.info(f"Predictions plotted and saved: {output_file}")


def main():
    """Main execution function for model training and prediction."""
    try:
        # Configuration flags
        import_data = True  # Set to True to fetch fresh data
        import_regressor = True  # Set to False to train new models

        # Load or import data
        data = load_or_import_data(import_data)

        # Process data
        logger.info("Processing sales data...")
        clean_data = process_sales_data(data)

        logger.info("Creating time series features...")
        feature_data = create_time_series_features(clean_data)
        logger.info("Data processing complete!")

        # Generate sample time series plot
        generate_sample_timeseries_plot(clean_data)

        # Load or train models
        load_or_train_models(import_regressor, feature_data)

        # Interactive prediction visualization
        interactive_prediction_visualization(feature_data)

    except Exception as e:
        logger.error(f"Script failed with error: {e}")
        raise
    finally:
        db_manager.close_connection()


if __name__ == "__main__":
    main()
