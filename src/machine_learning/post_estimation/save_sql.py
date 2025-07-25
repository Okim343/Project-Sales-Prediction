import urllib.parse
from sqlalchemy import create_engine
import pandas as pd


def get_connection_engine():
    """
    Creates and returns a SQLAlchemy engine for connecting to the remote PostgreSQL database.
    """
    user = "postgres"
    password = "Tommy627!"
    host = "172.27.40.210"
    port = "5432"
    dbname = "Mercado Livre"
    # Encode the password to handle special characters
    password_encoded = urllib.parse.quote_plus(password)
    engine = create_engine(
        f"postgresql://{user}:{password_encoded}@{host}:{port}/{dbname}"
    )
    return engine


def save_forecasts_to_sql(
    forecasts: dict, table_name: str = "sku_forecasts", if_exists: str = "replace"
):
    """
    Save forecast data for each SKU in the provided dictionary to the remote PostgreSQL database.
    Each SKU's forecast is a DataFrame with forecasted dates as the index and a 'prediction' column.
    This function concatenates all SKU forecasts into a single DataFrame with an additional 'sku' column,
    then writes the resulting table to the SQL database.

    Args:
        forecasts (dict): Dictionary with SKU as keys and forecast DataFrames as values.
        table_name (str, optional): The name of the SQL table to write to. Defaults to "sku_forecasts".
        if_exists (str, optional): Behavior if the table already exists. Defaults to "replace".
                                   Options: 'fail', 'replace', 'append'.
    """
    df_list = []
    for sku, forecast_df in forecasts.items():
        df = forecast_df.copy()
        df["sku"] = sku
        # Reset the index so that the date becomes a column
        df = df.reset_index().rename(columns={"index": "date"})
        df_list.append(df)

    # Combine all SKU forecasts into one DataFrame
    if df_list:
        final_df = pd.concat(df_list, ignore_index=True)
    else:
        final_df = pd.DataFrame(columns=["date", "prediction", "sku"])

    # Create a connection engine and write the DataFrame to SQL
    engine = get_connection_engine()
    final_df.to_sql(table_name, engine, if_exists=if_exists, index=False)  # type: ignore
