from sqlalchemy import create_engine
import pandas as pd


if __name__ == "__main__":
    print("Importing data from SQL...")
    user = "postgres"
    password = "Tommy627!"
    host = "172.27.240.181"
    port = "5432"
    dbname = "Mercado Livre"

    engine = create_engine(f"postgresql://{user}:{password}@{host}:{port}/{dbname}")
    query = "SELECT * FROM vw_orders_items"
    chunks = pd.read_sql_query(query, engine, chunksize=10000)
    df = pd.concat(chunks, ignore_index=True)

    print(df.head())
    print("Data imported successfully!")
