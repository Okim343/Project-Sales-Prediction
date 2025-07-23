"""
Database utilities module for centralized database operations.
Provides connection management, data import/export, and error handling.
"""

import logging
import pandas as pd
from typing import Optional
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from config import DatabaseConfig

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Centralized database operations manager."""

    def __init__(self):
        self.config = DatabaseConfig()
        self._engine: Optional[Engine] = None

    @property
    def engine(self) -> Engine:
        """Get or create database engine with connection pooling."""
        if self._engine is None:
            try:
                self._engine = create_engine(
                    self.config.get_connection_string(),
                    pool_pre_ping=True,  # Verify connections before use
                    pool_recycle=3600,  # Recycle connections after 1 hour
                )
                logger.info("Database connection established successfully")
            except SQLAlchemyError as e:
                logger.error(f"Failed to create database engine: {e}")
                raise
        return self._engine

    def import_data_from_sql(
        self, view: Optional[str] = None, chunksize: int = 10000
    ) -> pd.DataFrame:
        """
        Import data from PostgreSQL database with improved error handling.

        Args:
            view: Database view name (defaults to config value)
            chunksize: Number of rows to read per chunk

        Returns:
            DataFrame containing the imported data

        Raises:
            SQLAlchemyError: If database connection or query fails
        """
        view = view or self.config.VIEW

        try:
            logger.info(f"Importing data from {view}...")
            query = f"SELECT * FROM {view}"

            # Read data in chunks to handle large datasets
            chunks = pd.read_sql_query(query, self.engine, chunksize=chunksize)
            df = pd.concat(chunks, ignore_index=True)

            # Log memory usage information
            memory_usage_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
            logger.info(
                f"Data imported successfully. Shape: {df.shape}, Memory: {memory_usage_mb:.2f} MB"
            )

            return df

        except SQLAlchemyError as e:
            logger.error(f"Failed to import data from {view}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during data import: {e}")
            raise

    def save_forecasts_to_sql(
        self, forecasts: dict, table: Optional[str] = None
    ) -> None:
        """
        Save forecast data to PostgreSQL database.

        Args:
            forecasts: Dictionary of forecast DataFrames keyed by SKU
            table: Target table name (defaults to config value)

        Raises:
            SQLAlchemyError: If database save operation fails
        """
        table = table or self.config.FORECAST_TABLE

        try:
            logger.info(f"Saving forecasts to {table}...")

            # Combine all forecast DataFrames
            all_forecasts = []
            for sku, forecast_df in forecasts.items():
                if forecast_df is not None and not forecast_df.empty:
                    forecast_df = forecast_df.copy()
                    forecast_df["sku"] = sku
                    # Reset the index so that the date becomes a column (crucial step!)
                    forecast_df = forecast_df.reset_index().rename(
                        columns={"index": "date"}
                    )
                    all_forecasts.append(forecast_df)

            if not all_forecasts:
                logger.warning("No forecast data to save")
                return

            combined_df = pd.concat(all_forecasts, ignore_index=True)

            # Save to database with error handling
            combined_df.to_sql(
                table.split(".")[-1],  # Remove schema prefix for to_sql
                self.engine,
                schema=table.split(".")[0] if "." in table else None,
                if_exists="replace",
                index=False,
            )

            logger.info(
                f"Successfully saved {len(combined_df)} forecast records to {table}"
            )

        except SQLAlchemyError as e:
            logger.error(f"Failed to save forecasts to {table}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during forecast save: {e}")
            raise

    def test_connection(self) -> bool:
        """
        Test database connection.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("Database connection test successful")
            return True
        except SQLAlchemyError as e:
            logger.error(f"Database connection test failed: {e}")
            return False

    def close_connection(self) -> None:
        """Close database connection and clean up resources."""
        if self._engine:
            self._engine.dispose()
            self._engine = None
            logger.info("Database connection closed")


# Convenience functions for backward compatibility
def import_data_from_sql(
    user: str,
    password: str,
    host: str,
    port: str,
    dbname: str,
    view: str,
    chunksize: int = 10000,
) -> pd.DataFrame:
    """
    Legacy function for importing data (maintained for backward compatibility).
    Consider using DatabaseManager.import_data_from_sql() instead.
    """
    # Create temporary config for legacy calls
    original_config = DatabaseConfig()
    DatabaseConfig.USER = user
    DatabaseConfig.PASSWORD = password
    DatabaseConfig.HOST = host
    DatabaseConfig.PORT = port
    DatabaseConfig.DBNAME = dbname

    try:
        db_manager = DatabaseManager()
        return db_manager.import_data_from_sql(view, chunksize)
    finally:
        # Restore original config
        DatabaseConfig.USER = original_config.USER
        DatabaseConfig.PASSWORD = original_config.PASSWORD
        DatabaseConfig.HOST = original_config.HOST
        DatabaseConfig.PORT = original_config.PORT
        DatabaseConfig.DBNAME = original_config.DBNAME


# Global database manager instance
db_manager = DatabaseManager()
