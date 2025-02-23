from sqlalchemy import create_engine
import pandas as pd


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

    # Check and print the size of the DataFrame
    rows, cols = df.shape
    memory_usage_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    print(f"DataFrame shape: {rows} rows x {cols} columns")
    print(f"Total memory usage: {memory_usage_mb:.2f} MB")

    return df


def get_row_count(user, password, host, port, dbname, view):
    engine = create_engine(f"postgresql://{user}:{password}@{host}:{port}/{dbname}")
    count_query = f"SELECT COUNT(*) FROM {view}"
    with engine.connect() as conn:
        row_count = conn.execute(count_query).scalar()
    return row_count


if __name__ == "__main__":
    print("Importing data from SQL...")

    # Connection details
    user = "postgres"
    password = "Tommy627!"
    host = "172.27.240.181"
    port = "5432"
    dbname = "Mercado Livre"
    view = "public.vw_orders_items"

    rows_expected = get_row_count(user, password, host, port, dbname, view)
    print(f"Expected number of rows: {rows_expected}")

    df = import_data_from_sql(user, password, host, port, dbname, view)
    print("Data imported successfully!")
