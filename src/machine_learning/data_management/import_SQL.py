from sqlalchemy import create_engine
import pandas as pd
import logging

from data_management.config_ml import DATA

logger = logging.getLogger(__name__)


def import_data_from_sql(user, password, host, port, dbname, view, chunksize=10000):
    """
    Connects to a PostgreSQL database using SQLAlchemy and imports data using the specified SQL query.
    The data is loaded in chunks and concatenated into a single Pandas DataFrame.
    After importing, it prints the DataFrame's shape and memory usage.

    Parameters:
        user (str): Database username.
        password (str): Database password.
        host (str): Database host IP or hostname.
        port (str or int): Database port.
        dbname (str): Database name.
        view (str): What view is being called.
        chunksize (int): Number of rows to read per chunk (default is 10000).

    Returns:
        pd.DataFrame: DataFrame containing the imported data.
    """
    engine = create_engine(f"postgresql://{user}:{password}@{host}:{port}/{dbname}")
    query = f"SELECT * FROM {view}"
    chunks = pd.read_sql_query(query, engine, chunksize=chunksize)
    df = pd.concat(chunks, ignore_index=True)

    # Check the size of the DataFrame
    memory_usage_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    print(f"Total memory usage: {memory_usage_mb:.2f} MB")

    return df


def import_data_from_sql_since_date(
    user: str,
    password: str,
    host: str,
    port: str,
    dbname: str,
    view: str,
    since_date: str,
    chunksize: int = 10000,
) -> pd.DataFrame:
    """
    Import data from PostgreSQL database since a specific date.
    Enables fetching only new data while keeping full import option.

    Parameters:
        user (str): Database username.
        password (str): Database password.
        host (str): Database host IP or hostname.
        port (str or int): Database port.
        dbname (str): Database name.
        view (str): What view is being called.
        since_date (str): Date filter in YYYY-MM-DD format. Only records after this date will be imported.
        chunksize (int): Number of rows to read per chunk (default is 10000).

    Returns:
        pd.DataFrame: DataFrame containing the imported data since the specified date.

    Note:
        This function assumes the view has a 'date_created' column for filtering.
        Uses the existing import function as template but adds date filtering.
    """
    try:
        engine = create_engine(f"postgresql://{user}:{password}@{host}:{port}/{dbname}")

        # Add date filter to query - only get records after since_date
        query = f"SELECT * FROM {view} WHERE date_created > '{since_date}' ORDER BY date_created DESC"

        logger.info(f"Importing data from {view} since {since_date}")

        # Read data in chunks with date filter
        chunks = pd.read_sql_query(query, engine, chunksize=chunksize)
        df = pd.concat(chunks, ignore_index=True)

        # Check the size of the DataFrame
        memory_usage_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        logger.info(f"Imported {len(df)} records since {since_date}")
        logger.info(f"Total memory usage: {memory_usage_mb:.2f} MB")

        return df

    except Exception as e:
        logger.error(f"Failed to import data since {since_date}: {e}")
        # Return empty DataFrame with expected columns if import fails
        return pd.DataFrame()


if __name__ == "__main__":
    print("Importing data from SQL...")

    # Connection details
    user = "postgres"
    password = "Tommy627!"
    host = "172.27.40.210"
    port = "5432"
    dbname = "Mercado Livre"
    view = "public.view_enrico"
    produces = DATA / "raw_sql.csv"

    data = import_data_from_sql(user, password, host, port, dbname, view)
    data.to_csv(produces, index=False)

    print("Data imported successfully!")
