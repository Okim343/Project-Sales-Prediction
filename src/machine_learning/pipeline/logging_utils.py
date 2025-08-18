"""
Logging utilities for pipeline runners to provide consistent, clean, and readable logging.

This module contains reusable logging functions that help reduce verbosity and repetition
while maintaining all important information for monitoring and debugging.
"""

import logging
from typing import Dict, List, Optional, Any


def get_pipeline_logger(name: str = __name__) -> logging.Logger:
    """Get a properly configured logger for pipeline operations."""
    return logging.getLogger(name)


def log_pipeline_start(
    mode: str, logger: logging.Logger, version: str = "", extra_info: str = ""
) -> None:
    """
    Log the start of a pipeline with consistent formatting.

    Args:
        mode: Pipeline mode (daily, full, monthly, since-date)
        logger: Logger instance to use
        version: Optional version suffix (e.g., "test")
        extra_info: Optional additional information (e.g., since date)
    """
    mode_display = f"{mode.upper()}{' ' + version.upper() if version else ''}"
    info_display = f" ({extra_info})" if extra_info else ""
    separator = "=" * (
        len(mode_display) + len(" MODE PIPELINE STARTED") + len(info_display)
    )

    logger.info(separator)
    logger.info(f"{mode_display} MODE PIPELINE STARTED{info_display}")
    logger.info(separator)


def log_data_info(
    records: int,
    logger: logging.Logger,
    date_range: Optional[Dict] = None,
    operation: str = "imported",
) -> None:
    """
    Log data import/processing information in a consistent format.

    Args:
        records: Number of records processed
        logger: Logger instance to use
        date_range: Optional dict with min_date, max_date, date_span_days keys
        operation: Description of the operation (imported, processed, etc.)
    """
    if date_range and date_range.get("min_date") and date_range.get("max_date"):
        date_info = (
            f" from {date_range['min_date']} to {date_range['max_date']} "
            f"({date_range['date_span_days']} days)"
        )
    else:
        date_info = ""

    logger.info(f"Data {operation}: {records:,} records{date_info}")


def log_model_operations(
    operation: str,
    count: int,
    logger: logging.Logger,
    details: str = "",
    level: str = "info",
) -> None:
    """
    Log model operations (loading, saving, updating) with consistent formatting.

    Args:
        operation: Type of operation (loaded, saved, updated, etc.)
        count: Number of models affected
        logger: Logger instance to use
        details: Optional additional details
        level: Log level (info, debug, warning)
    """
    message = f"Models {operation}: {count:,}"
    if details:
        message += f" - {details}"

    getattr(logger, level.lower())(message)


def log_rollback_summary(
    stats: Dict[str, int], logger: logging.Logger, context: str = ""
) -> None:
    """
    Log rollback statistics in a concise, readable format.

    Args:
        stats: Dictionary with rollback statistics
        logger: Logger instance to use
        context: Optional context for the statistics (e.g., "test mode")
    """
    total = stats.get("total_processed", 0)
    improved = stats.get("models_improved", 0)
    maintained = stats.get("models_maintained", 0)
    rolled_back = stats.get("models_rolled_back", 0)
    failed = stats.get("mlbs_failed", 0)

    context_suffix = f" ({context})" if context else ""

    logger.info(f"Processing Results{context_suffix}:")
    logger.info(
        f"  Total processed: {total} | Improved: {improved} | Maintained: {maintained}"
    )
    logger.info(f"  Rolled back: {rolled_back} | Failed: {failed}")

    if total > 0:
        success_rate = ((improved + maintained) / total) * 100
        logger.info(f"  Success rate: {success_rate:.1f}%")


def log_validation_summary(
    issues: List[str],
    model_count: int,
    logger: logging.Logger,
    context: str = "",
    max_issues: int = 5,
) -> None:
    """
    Log validation results in a concise format, showing only key issues.

    Args:
        issues: List of validation issues found
        model_count: Number of models validated
        logger: Logger instance to use
        context: Optional context (e.g., "final validation", "test mode")
        max_issues: Maximum number of issues to display in detail
    """
    context_suffix = f" ({context})" if context else ""

    if not issues:
        logger.info(
            f"Validation complete{context_suffix}: All {model_count} models passed"
        )
        return

    logger.warning(f"Validation found {len(issues)} issues{context_suffix}:")

    # Log first few issues in detail
    for issue in issues[:max_issues]:
        logger.warning(f"  - {issue}")

    # Summarize remaining issues
    if len(issues) > max_issues:
        logger.warning(f"  ... and {len(issues) - max_issues} more issues")


def log_pipeline_completion(
    status: str,
    execution_time: float,
    metrics: Dict[str, Any],
    logger: logging.Logger,
    mode: str,
    version: str = "",
) -> None:
    """
    Log pipeline completion with standardized summary information.

    Args:
        status: Pipeline completion status (success, partial, failed)
        execution_time: Total execution time in seconds
        metrics: Dictionary with key metrics (records_processed, models_updated, etc.)
        logger: Logger instance to use
        mode: Pipeline mode
        version: Optional version suffix
    """
    mode_display = f"{mode.upper()}{' ' + version.upper() if version else ''}"
    status_display = status.upper()

    # Choose appropriate log level based on status
    log_func = (
        logger.info
        if status == "success"
        else logger.warning
        if status == "partial"
        else logger.error
    )

    log_func(f"{mode_display} MODE PIPELINE COMPLETED - {status_display}")
    logger.info(f"Execution time: {execution_time:.1f} seconds")

    # Log key metrics
    records = metrics.get("records_processed", 0)
    models = metrics.get("models_updated", 0)

    logger.info(f"Results: {records:,} records processed, {models} models updated")

    # Log any error details for non-success statuses
    error_message = metrics.get("error_message")
    if error_message and status != "success":
        logger.warning(f"Issues: {error_message}")


def log_memory_usage(
    memory_stats: Dict[str, float],
    logger: logging.Logger,
    context: str = "",
    threshold_mb: float = 1000,
) -> None:
    """
    Log memory usage information, but only when it exceeds threshold or explicitly requested.

    Args:
        memory_stats: Dictionary with memory statistics (rss_mb, percent)
        logger: Logger instance to use
        context: Optional context for the memory check
        threshold_mb: Only log if memory usage exceeds this threshold
    """
    rss_mb = memory_stats.get("rss_mb", 0)
    percent = memory_stats.get("percent", 0)

    # Only log if memory usage is high or context is explicitly provided
    if rss_mb > threshold_mb or context:
        context_suffix = f" during {context}" if context else ""
        logger.debug(
            f"Memory usage{context_suffix}: {rss_mb:.1f} MB RSS ({percent:.1f}% of system)"
        )


def log_batch_progress(
    batch_num: int,
    total_batches: int,
    items_range: str,
    logger: logging.Logger,
    context: str = "",
) -> None:
    """
    Log batch processing progress in a consistent format.

    Args:
        batch_num: Current batch number (1-indexed)
        total_batches: Total number of batches
        items_range: Description of items in this batch (e.g., "MLBs 1-5")
        logger: Logger instance to use
        context: Optional context (e.g., "test batch")
    """
    context_prefix = f"{context.title()} " if context else ""
    logger.info(f"{context_prefix}Batch {batch_num}/{total_batches}: {items_range}")


def log_database_operation(
    operation: str,
    count: int,
    logger: logging.Logger,
    status: str = "completed",
    details: str = "",
) -> None:
    """
    Log database operations (save, load) with consistent formatting.

    Args:
        operation: Type of operation (saving, loading)
        count: Number of items affected
        logger: Logger instance to use
        status: Operation status (completed, failed, skipped)
        details: Optional additional details
    """
    if status == "skipped":
        logger.info(f"Database {operation}: {status} - {details}")
    elif status == "failed":
        logger.error(f"Database {operation} failed: {details}")
    else:
        message = f"Database {operation}: {count:,} items {status}"
        if details:
            message += f" - {details}"
        logger.info(message)


def log_model_comparison_results(
    comparison_results: Dict, logger: logging.Logger, context: str = ""
) -> None:
    """
    Log model comparison results in a concise format.

    Args:
        comparison_results: Dictionary with comparison results and summary
        logger: Logger instance to use
        context: Optional context (e.g., "monthly retraining")
    """
    if "model_comparison" not in comparison_results:
        return

    summary = comparison_results["model_comparison"].get("summary", {})
    context_suffix = f" ({context})" if context else ""

    logger.info(f"Model Comparison Results{context_suffix}:")

    if "avg_improvement_pct" in summary:
        logger.info(f"  Average improvement: {summary['avg_improvement_pct']:.2f}%")

    if "comparison_errors" in summary:
        errors = summary["comparison_errors"]
        if errors > 0:
            logger.warning(f"  Comparison errors: {errors}")


def log_configuration_check(
    config_items: Dict[str, Any], logger: logging.Logger
) -> None:
    """
    Log important configuration settings at pipeline start.

    Args:
        config_items: Dictionary of configuration items to log
        logger: Logger instance to use
    """
    logger.debug("Pipeline Configuration:")
    for key, value in config_items.items():
        logger.debug(f"  {key}: {value}")
