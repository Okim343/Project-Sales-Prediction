"""
Production logging configuration with file rotation and multiple outputs.

Provides structured logging suitable for production monitoring and debugging
while maintaining compatibility with existing logging patterns.
"""

import os
import logging
import logging.handlers
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from .path_resolver import resolve_project_paths


class ProductionLoggingFormatter(logging.Formatter):
    """Custom formatter for production logging with enhanced context."""

    def __init__(self):
        # Use the same format as existing codebase but with additional context
        super().__init__(
            fmt="%(asctime)s - %(name)s - %(levelname)s - [%(process)d] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    def format(self, record):
        # Add deployment context if available
        if not hasattr(record, "deployment_mode"):
            record.deployment_mode = getattr(self, "_deployment_mode", "unknown")

        return super().format(record)


class ProductionLoggingConfig:
    """Centralized production logging configuration."""

    def __init__(self, mode: str = "production"):
        self.mode = mode
        self.log_level = self._get_log_level()
        self.log_dir = self._get_log_directory()
        self._configured_loggers: Dict[str, logging.Logger] = {}

    def _get_log_level(self) -> int:
        """Get logging level based on environment and mode."""
        # Check environment variable first
        env_level = os.getenv("LOG_LEVEL", "").upper()
        if env_level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            return getattr(logging, env_level)

        # Default levels by mode
        mode_levels = {
            "production": logging.INFO,
            "development": logging.DEBUG,
            "testing": logging.WARNING,
        }

        return mode_levels.get(self.mode, logging.INFO)

    def _get_log_directory(self) -> Path:
        """Get log directory, creating if necessary."""
        try:
            paths = resolve_project_paths()
            log_dir = paths.get("logs", paths["project_root"] / "logs")
        except Exception:
            # Fallback if path resolution fails
            log_dir = Path.cwd() / "logs"

        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir

    def _create_file_handler(
        self,
        log_filename: str,
        max_bytes: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
    ) -> logging.handlers.RotatingFileHandler:
        """Create rotating file handler with production settings."""
        log_file_path = self.log_dir / log_filename

        handler = logging.handlers.RotatingFileHandler(
            filename=log_file_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )

        handler.setFormatter(ProductionLoggingFormatter())
        handler.setLevel(self.log_level)

        return handler

    def _create_console_handler(self) -> logging.StreamHandler:
        """Create console handler with appropriate formatting."""
        handler = logging.StreamHandler()
        handler.setFormatter(ProductionLoggingFormatter())

        # Console can be more verbose for debugging
        console_level = min(self.log_level, logging.INFO)
        handler.setLevel(console_level)

        return handler

    def setup_logger(
        self,
        name: str,
        log_filename: Optional[str] = None,
        include_console: bool = True,
    ) -> logging.Logger:
        """
        Setup a logger with production configuration.

        Args:
            name: Logger name (typically __name__)
            log_filename: Custom log filename (optional)
            include_console: Whether to include console output

        Returns:
            Configured logger instance
        """
        # Return existing logger if already configured
        if name in self._configured_loggers:
            return self._configured_loggers[name]

        logger = logging.getLogger(name)

        # Clear any existing handlers to avoid duplication
        logger.handlers.clear()
        logger.setLevel(self.log_level)

        # Create log filename if not provided
        if log_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d")
            log_filename = f"pipeline_{self.mode}_{timestamp}.log"

        # Add file handler
        file_handler = self._create_file_handler(log_filename)
        logger.addHandler(file_handler)

        # Add console handler if requested
        if include_console:
            console_handler = self._create_console_handler()
            logger.addHandler(console_handler)

        # Store formatter mode for context
        for handler in logger.handlers:
            if hasattr(handler.formatter, "_deployment_mode"):
                handler.formatter._deployment_mode = self.mode

        # Cache logger
        self._configured_loggers[name] = logger

        logger.info(
            f"Production logging configured for {name} - Level: {logging.getLevelName(self.log_level)}"
        )

        return logger

    def setup_root_logger(self) -> logging.Logger:
        """Setup root logger for the entire application."""
        return self.setup_logger(
            name="deploy_pipeline",
            log_filename=f"deploy_pipeline_{datetime.now().strftime('%Y%m%d')}.log",
            include_console=True,
        )

    def get_pipeline_logger(self, pipeline_mode: str) -> logging.Logger:
        """Get logger specifically configured for pipeline operations."""
        logger_name = f"pipeline.{pipeline_mode}"
        log_filename = (
            f"pipeline_{pipeline_mode}_{datetime.now().strftime('%Y%m%d')}.log"
        )

        return self.setup_logger(
            name=logger_name, log_filename=log_filename, include_console=True
        )


# Global configuration instance
_logging_config: Optional[ProductionLoggingConfig] = None


def setup_production_logging(
    mode: str = "production", pipeline_mode: str = "daily"
) -> logging.Logger:
    """
    Setup production logging configuration.

    Args:
        mode: Deployment mode (production, development, testing)
        pipeline_mode: Pipeline mode (daily, full, monthly, since-date)

    Returns:
        Root logger configured for production use
    """
    global _logging_config

    try:
        _logging_config = ProductionLoggingConfig(mode=mode)

        # Setup root logger
        root_logger = _logging_config.setup_root_logger()

        # Also setup pipeline-specific logger
        pipeline_logger = _logging_config.get_pipeline_logger(pipeline_mode)  # noqa: F841

        root_logger.info(
            f"Production logging initialized - Mode: {mode}, Pipeline: {pipeline_mode}"
        )
        root_logger.info(f"Log directory: {_logging_config.log_dir}")

        return root_logger

    except Exception as e:
        # Fallback to basic logging if production setup fails
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        logger = logging.getLogger("deploy_pipeline.fallback")
        logger.error(f"Failed to setup production logging, using fallback: {e}")
        return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with production configuration.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance
    """
    global _logging_config

    if _logging_config is None:
        # Initialize with default settings if not already configured
        setup_production_logging()

    return _logging_config.setup_logger(name)


def configure_existing_loggers() -> None:
    """
    Configure existing loggers to use production settings.

    This ensures that loggers from the ML pipeline also use production formatting.
    """
    global _logging_config

    if _logging_config is None:
        return

    # Get all existing loggers
    existing_loggers = [
        logging.getLogger(name) for name in logging.root.manager.loggerDict
    ]

    # Configure pipeline loggers to use our format
    pipeline_logger_prefixes = [
        "src.machine_learning",
        "machine_learning",
        "config",
        "database_utils",
        "data_management",
        "estimation",
        "validation",
        "pipeline",
    ]

    for logger in existing_loggers:
        if any(logger.name.startswith(prefix) for prefix in pipeline_logger_prefixes):
            # Don't add handlers if they already exist
            if not logger.handlers:
                # Add file handler only (console will be handled by root)
                file_handler = _logging_config._create_file_handler(
                    f"ml_components_{datetime.now().strftime('%Y%m%d')}.log"
                )
                logger.addHandler(file_handler)
                logger.setLevel(_logging_config.log_level)


def cleanup_logging() -> None:
    """Cleanup logging handlers and close files."""
    global _logging_config

    if _logging_config is None:
        return

    for logger in _logging_config._configured_loggers.values():
        for handler in logger.handlers[
            :
        ]:  # Copy list to avoid modification during iteration
            if isinstance(handler, logging.FileHandler):
                handler.close()
            logger.removeHandler(handler)

    _logging_config._configured_loggers.clear()
    _logging_config = None


def log_deployment_info(logger: logging.Logger, **context: Any) -> None:
    """
    Log deployment context information.

    Args:
        logger: Logger instance to use
        **context: Key-value pairs of deployment context
    """
    logger.info("=== DEPLOYMENT CONTEXT ===")
    for key, value in context.items():
        logger.info(f"{key}: {value}")
    logger.info("=" * 25)
