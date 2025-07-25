"""
Centralized configuration module for the sales forecasting application.
Handles environment variables, database connections, and application settings.
"""

import os
from pathlib import Path
from typing import Optional
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

# Environment variables are loaded from the system environment
# You can set these in your shell or system environment:
# export DB_USER=postgres
# export DB_PASSWORD=your_password
# etc.

# Path Configuration
SRC = Path(__file__).parent.resolve()
ROOT = SRC.parent.parent.resolve()
BLD = ROOT / "bld"
DATA = ROOT / "data"

# Ensure directories exist
BLD.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)


# Database Configuration
class DatabaseConfig:
    """Database configuration and connection management."""

    USER = os.getenv("DB_USER", "postgres")
    PASSWORD = os.getenv(
        "DB_PASSWORD", "Tommy627!"
    )  # Default for backward compatibility
    HOST = os.getenv("DB_HOST", "172.27.40.210")
    PORT = os.getenv("DB_PORT", "5432")
    DBNAME = os.getenv("DB_NAME", "Mercado Livre")
    VIEW = os.getenv("DB_VIEW", "public.view_enrico")
    FORECAST_TABLE = os.getenv("DB_FORECAST_TABLE", "public.mlb_forecasts_90_days")

    _engine: Optional[Engine] = None

    @classmethod
    def get_connection_string(cls) -> str:
        """Get the database connection string."""
        return (
            f"postgresql://{cls.USER}:{cls.PASSWORD}@{cls.HOST}:{cls.PORT}/{cls.DBNAME}"
        )

    @classmethod
    def get_engine(cls) -> Engine:
        """Get or create a SQLAlchemy engine instance."""
        if cls._engine is None:
            cls._engine = create_engine(cls.get_connection_string())
        return cls._engine


# Application Settings
class AppConfig:
    """Application-specific configuration."""

    FORECAST_DAYS = int(os.getenv("FORECAST_DAYS", "30"))
    FORECAST_DAYS_LONG = int(os.getenv("FORECAST_DAYS_LONG", "90"))
    DEFAULT_SKU = os.getenv("DEFAULT_SKU", "TC213")
    DEFAULT_MLB = os.getenv(
        "DEFAULT_MLB", "TC213"
    )  # Will update once we know MLB codes
    ACTIVE_MLB_DAYS_THRESHOLD = int(os.getenv("ACTIVE_MLB_DAYS_THRESHOLD", "30"))

    # Model Features
    MODEL_FEATURES = ["day_of_week", "day_of_month", "rolling_mean_3", "lag_1", "price"]
    TARGET_COLUMN = "quant"

    # File Paths
    RAW_DATA_FILE = DATA / "raw_sql.csv"
    FEATURE_DATA_FILE = BLD / "feature_data.csv"
    MLB_REGRESSORS_FILE = BLD / "mlb_regressors.pkl"
    MLB_FORECAST_FILE = BLD / "mlb_forecast.pkl"
    # Keep legacy names for backward compatibility
    SKU_REGRESSORS_FILE = BLD / "mlb_regressors.pkl"  # Alias for backward compatibility
    SKU_FORECAST_FILE = BLD / "mlb_forecast.pkl"  # Alias for backward compatibility
