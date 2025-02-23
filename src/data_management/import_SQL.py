from sqlalchemy import create_engine
import pandas as pd

from config import BLD


def import_data_from_sql(user, password, host, port, dbname, query, chunksize=10000):
    """
    Connects to a PostgreSQL database using SQLAlchemy and imports data
    using the specified SQL query.
    The data is loaded in chunks and concatenated into a single Pandas DataFrame.

    Parameters:
        user (str): Database username.
        password (str): Database password.
        host (str): Database host IP or hostname.
        port (str or int): Database port.
        dbname (str): Database name.
        query (str): SQL query to execute.
        chunksize (int): Number of rows to read per chunk (default is 10000).

    Returns:
        pd.DataFrame: DataFrame containing the imported data.
    """
    engine = create_engine(f"postgresql://{user}:{password}@{host}:{port}/{dbname}")
    chunks = pd.read_sql_query(query, engine, chunksize=chunksize)
    df = pd.concat(chunks, ignore_index=True)

    return df


if __name__ == "__main__":
    print("Importing data from SQL...")

    user = "postgres"
    password = "Tommy627!"
    host = "172.27.40.210"
    port = "5432"
    dbname = "Mercado Livre"
    query = "SELECT * FROM public.vw_orders_items"

    data = import_data_from_sql(user, password, host, port, dbname, query)
    produce = BLD / "rawdata_SQL.csv"
    data.to_csv(produce, index=True)
    print("Data imported and saved successfully!")
