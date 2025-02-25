from sqlalchemy import create_engine
import pandas as pd

from config_ml import DATA


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
    query = f"SELECT COUNT(*) FROM {view}"
    chunks = pd.read_sql_query(query, engine, chunksize=chunksize)
    df = pd.concat(chunks, ignore_index=True)

    # Check the size of the DataFrame
    memory_usage_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    print(f"Total memory usage: {memory_usage_mb:.2f} MB")

    return df


if __name__ == "__main__":
    print("Importing data from SQL...")

    # Connection details
    user = "postgres"
    password = "Tommy627!"
    host = "172.27.40.210"
    port = "5432"
    dbname = "Mercado Livre"
    view = "public.vw_orders_items"
    produces = DATA / "raw" / "raw_sql.csv"

    data = import_data_from_sql(user, password, host, port, dbname, view)
    data.to_csv(produces, index=False)

    print("Data imported successfully!")
