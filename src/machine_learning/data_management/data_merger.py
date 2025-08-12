"""
Data merging module for the continuous learning pipeline.
Handles merging new data with historical data and deduplication.
"""

import logging
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)


def merge_with_historical(
    new_data: pd.DataFrame, historical_path: Path
) -> pd.DataFrame:
    """
    Merge new data with historical data from a file.
    Load historical data, append new data, and remove duplicates based on order_id.

    Args:
        new_data: DataFrame containing new records to merge
        historical_path: Path to historical data file (CSV)

    Returns:
        DataFrame containing merged data with duplicates removed

    Note:
        - Handles edge cases: file not exists, empty new data
        - Uses 'order_id' column for deduplication (assumes it exists)
        - Sorts final data by date_created in descending order
    """
    try:
        # Handle empty new data case
        if new_data.empty:
            logger.info("No new data to merge")
            if historical_path.exists():
                logger.info(f"Loading existing historical data from {historical_path}")
                return pd.read_csv(historical_path)
            else:
                logger.warning(
                    "No historical data file exists and no new data provided"
                )
                return pd.DataFrame()

        # Load historical data if file exists
        if historical_path.exists():
            logger.info(f"Loading historical data from {historical_path}")
            historical_data = pd.read_csv(historical_path)

            if historical_data.empty:
                logger.info("Historical data file is empty, using only new data")
                merged_data = new_data.copy()
            else:
                logger.info(
                    f"Merging {len(new_data)} new records with {len(historical_data)} historical records"
                )

                # Concatenate historical and new data
                merged_data = pd.concat([historical_data, new_data], ignore_index=True)

                # Remove duplicates based on order_id (keep last occurrence - new data preferred)
                if "order_id" in merged_data.columns:
                    initial_count = len(merged_data)
                    merged_data = merged_data.drop_duplicates(
                        subset=["order_id"], keep="last"
                    )
                    duplicates_removed = initial_count - len(merged_data)

                    if duplicates_removed > 0:
                        logger.info(
                            f"Removed {duplicates_removed} duplicate records based on order_id"
                        )
                else:
                    logger.warning("No 'order_id' column found for deduplication")
        else:
            logger.info("No historical data file exists, using only new data")
            merged_data = new_data.copy()

        # Sort by date_created if column exists
        if "date_created" in merged_data.columns:
            merged_data = merged_data.sort_values("date_created", ascending=False)
            logger.info("Sorted merged data by date_created (newest first)")

        logger.info(f"Final merged dataset contains {len(merged_data)} records")
        return merged_data

    except Exception as e:
        logger.error(
            f"Failed to merge data with historical file {historical_path}: {e}"
        )
        # Return new data if merge fails
        logger.warning("Returning only new data due to merge failure")
        return new_data.copy() if not new_data.empty else pd.DataFrame()


def save_merged_data(
    data: pd.DataFrame, output_path: Path, backup_existing: bool = True
) -> bool:
    """
    Save merged data to file with optional backup of existing file.

    Args:
        data: DataFrame to save
        output_path: Path where to save the data
        backup_existing: Whether to backup existing file before overwriting

    Returns:
        True if save was successful, False otherwise
    """
    try:
        # Create parent directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Backup existing file if requested
        if backup_existing and output_path.exists():
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = output_path.with_suffix(
                f".backup_{timestamp}{output_path.suffix}"
            )
            output_path.rename(backup_path)
            logger.info(f"Backed up existing file to {backup_path}")

        # Save the merged data
        data.to_csv(output_path, index=False)

        # Log memory usage information
        memory_usage_mb = data.memory_usage(deep=True).sum() / (1024 * 1024)
        logger.info(f"Saved {len(data)} records to {output_path}")
        logger.info(f"File size: {memory_usage_mb:.2f} MB")

        return True

    except Exception as e:
        logger.error(f"Failed to save merged data to {output_path}: {e}")
        return False


def get_date_range_info(data: pd.DataFrame) -> dict:
    """
    Get information about the date range in the dataset.

    Args:
        data: DataFrame to analyze

    Returns:
        Dictionary with date range information
    """
    try:
        # Check for date column in both processed and raw data formats
        date_column = None
        if "date" in data.columns:
            date_column = "date"
        elif "date_created" in data.columns:
            date_column = "date_created"

        if data.empty or date_column is None:
            # Check if date information is in the DataFrame index
            if isinstance(data.index, pd.DatetimeIndex) and not data.empty:
                min_date = data.index.min()
                max_date = data.index.max()
                date_span = (max_date - min_date).days

                return {
                    "min_date": min_date.strftime("%Y-%m-%d"),
                    "max_date": max_date.strftime("%Y-%m-%d"),
                    "date_span_days": date_span,
                    "record_count": len(data),
                }
            else:
                return {
                    "min_date": None,
                    "max_date": None,
                    "date_span_days": 0,
                    "record_count": len(data),
                }

        # Convert to datetime and handle timezone properly
        date_col = pd.to_datetime(data[date_column])
        if date_col.dt.tz is not None:
            # If timezone-aware, convert to UTC then remove timezone
            data[date_column] = date_col.dt.tz_convert("UTC").dt.tz_localize(None)
        else:
            # If timezone-naive, keep as is
            data[date_column] = date_col

        min_date = data[date_column].min()
        max_date = data[date_column].max()
        date_span = (max_date - min_date).days

        return {
            "min_date": min_date.strftime("%Y-%m-%d"),
            "max_date": max_date.strftime("%Y-%m-%d"),
            "date_span_days": date_span,
            "record_count": len(data),
        }

    except Exception as e:
        logger.error(f"Failed to analyze date range: {e}")
        return {
            "min_date": None,
            "max_date": None,
            "date_span_days": 0,
            "record_count": len(data),
        }
