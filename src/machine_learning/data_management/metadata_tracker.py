"""
Metadata tracking module for the continuous learning pipeline.
Tracks pipeline runs without disrupting current workflow.
"""

import logging
from datetime import datetime
from typing import Optional, Dict, Any
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from database_utils import db_manager

logger = logging.getLogger(__name__)


def create_metadata_table() -> bool:
    """
    Create the pipeline metadata table if it doesn't exist.

    Returns:
        True if table creation was successful or table already exists, False otherwise
    """
    create_table_query = """
    CREATE TABLE IF NOT EXISTS pipeline_metadata (
        run_id SERIAL PRIMARY KEY,
        run_timestamp TIMESTAMP,
        run_type VARCHAR(20),
        status VARCHAR(20),
        records_processed INT,
        models_updated INT,
        error_message TEXT,
        run_duration_seconds FLOAT
    );
    """

    try:
        with db_manager.engine.connect() as conn:
            conn.execute(text(create_table_query))
            conn.commit()
        logger.info("Pipeline metadata table created successfully or already exists")
        return True
    except SQLAlchemyError as e:
        logger.error(f"Failed to create metadata table: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error creating metadata table: {e}")
        return False


def log_pipeline_run(
    run_type: str,
    status: str,
    records_processed: int = 0,
    models_updated: int = 0,
    error_message: Optional[str] = None,
    run_duration_seconds: Optional[float] = None,
) -> bool:
    """
    Log pipeline run information to the metadata table.

    Args:
        run_type: Type of run ('daily', 'weekly', 'full', 'test')
        status: Run status ('success', 'failed', 'partial')
        records_processed: Number of data records processed
        models_updated: Number of models updated/trained
        error_message: Error message if status is 'failed'
        run_duration_seconds: Total execution time in seconds

    Returns:
        True if logging was successful, False otherwise
    """
    insert_query = """
    INSERT INTO pipeline_metadata (
        run_timestamp, run_type, status, records_processed,
        models_updated, error_message, run_duration_seconds
    ) VALUES (
        :run_timestamp, :run_type, :status, :records_processed,
        :models_updated, :error_message, :run_duration_seconds
    )
    """

    try:
        with db_manager.engine.connect() as conn:
            conn.execute(
                text(insert_query),
                {
                    "run_timestamp": datetime.now(),
                    "run_type": run_type,
                    "status": status,
                    "records_processed": records_processed,
                    "models_updated": models_updated,
                    "error_message": error_message,
                    "run_duration_seconds": run_duration_seconds,
                },
            )
            conn.commit()

        logger.info(f"Pipeline run logged: {run_type} - {status}")
        return True

    except SQLAlchemyError as e:
        logger.error(f"Failed to log pipeline run: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error logging pipeline run: {e}")
        return False


def get_last_successful_run(run_type: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Retrieve information about the last successful pipeline run.

    Args:
        run_type: Filter by specific run type ('daily', 'weekly', 'full', 'test').
                 If None, returns last successful run of any type.

    Returns:
        Dictionary with run information or None if no successful runs found
    """
    base_query = """
    SELECT run_id, run_timestamp, run_type, records_processed,
           models_updated, run_duration_seconds
    FROM pipeline_metadata
    WHERE status = 'success'
    """

    if run_type:
        query = base_query + " AND run_type = :run_type"
        params = {"run_type": run_type}
    else:
        query = base_query
        params = {}

    query += " ORDER BY run_timestamp DESC LIMIT 1"

    try:
        with db_manager.engine.connect() as conn:
            result = conn.execute(text(query), params).fetchone()

        if result:
            return {
                "run_id": result[0],
                "run_timestamp": result[1],
                "run_type": result[2],
                "records_processed": result[3],
                "models_updated": result[4],
                "run_duration_seconds": result[5],
            }
        else:
            logger.info(f"No successful runs found for type: {run_type or 'any'}")
            return None

    except SQLAlchemyError as e:
        logger.error(f"Failed to retrieve last successful run: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error retrieving last successful run: {e}")
        return None


def get_pipeline_stats(days_back: int = 30) -> Dict[str, Any]:
    """
    Get pipeline execution statistics for the last N days.

    Args:
        days_back: Number of days to look back for statistics

    Returns:
        Dictionary with pipeline statistics
    """
    query = """
    SELECT
        COUNT(*) as total_runs,
        COUNT(CASE WHEN status = 'success' THEN 1 END) as successful_runs,
        COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed_runs,
        AVG(run_duration_seconds) as avg_duration_seconds,
        MAX(run_duration_seconds) as max_duration_seconds,
        MIN(run_duration_seconds) as min_duration_seconds
    FROM pipeline_metadata
    WHERE run_timestamp >= NOW() - INTERVAL ':days_back days'
    """

    try:
        with db_manager.engine.connect() as conn:
            result = conn.execute(text(query), {"days_back": days_back}).fetchone()

        if result:
            return {
                "total_runs": result[0] or 0,
                "successful_runs": result[1] or 0,
                "failed_runs": result[2] or 0,
                "success_rate": (result[1] or 0) / max(result[0] or 1, 1) * 100,
                "avg_duration_seconds": result[3] or 0,
                "max_duration_seconds": result[4] or 0,
                "min_duration_seconds": result[5] or 0,
                "days_analyzed": days_back,
            }
        else:
            return {
                "total_runs": 0,
                "successful_runs": 0,
                "failed_runs": 0,
                "success_rate": 0,
                "avg_duration_seconds": 0,
                "max_duration_seconds": 0,
                "min_duration_seconds": 0,
                "days_analyzed": days_back,
            }

    except SQLAlchemyError as e:
        logger.error(f"Failed to retrieve pipeline statistics: {e}")
        return {}
    except Exception as e:
        logger.error(f"Unexpected error retrieving pipeline statistics: {e}")
        return {}
