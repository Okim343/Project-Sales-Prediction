"""
Deployment utilities for production-ready pipeline execution.

This package provides deployment-specific functionality while keeping
the core ML pipeline logic clean and focused.
"""

from .logging_config import setup_production_logging, log_deployment_info
from .environment_validator import validate_production_environment
from .cleanup_handler import (
    register_cleanup_handlers,
    cleanup_resources,
    deployment_context,
    emergency_cleanup,
)
from .path_resolver import resolve_project_paths, setup_working_directory

__all__ = [
    "setup_production_logging",
    "log_deployment_info",
    "validate_production_environment",
    "register_cleanup_handlers",
    "cleanup_resources",
    "deployment_context",
    "emergency_cleanup",
    "resolve_project_paths",
    "setup_working_directory",
]
