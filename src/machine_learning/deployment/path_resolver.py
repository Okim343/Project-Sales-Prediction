"""
Path resolution utilities for production deployment.

Handles absolute path resolution and working directory setup for cron execution
where the execution context may be different from interactive usage.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Tuple, Dict, Optional

logger = logging.getLogger(__name__)


class ProjectPathResolver:
    """Resolves project paths for production deployment contexts."""

    def __init__(self):
        self._project_root: Optional[Path] = None
        self._src_path: Optional[Path] = None
        self._ml_path: Optional[Path] = None

    def detect_project_root(self) -> Path:
        """
        Detect project root directory from current file location.

        Returns:
            Path to project root directory

        Raises:
            RuntimeError: If project root cannot be determined
        """
        if self._project_root is not None:
            return self._project_root

        # Start from current file and search upward for project markers
        current_path = Path(__file__).resolve()

        # Look for project markers (environment.yml, CLAUDE.md, src directory)
        project_markers = ["environment.yml", "CLAUDE.md", "README.md"]

        for parent in [current_path] + list(current_path.parents):
            # Check if this directory contains project markers
            has_markers = any((parent / marker).exists() for marker in project_markers)
            has_src_dir = (parent / "src").is_dir()

            if has_markers and has_src_dir:
                self._project_root = parent
                logger.info(f"Detected project root: {self._project_root}")
                return self._project_root

        # Fallback: use relative path detection
        try:
            # Assume we're in src/machine_learning/deployment/
            fallback_root = current_path.parent.parent.parent
            if (fallback_root / "src").is_dir():
                self._project_root = fallback_root
                logger.warning(
                    f"Using fallback project root detection: {self._project_root}"
                )
                return self._project_root
        except Exception as e:
            logger.error(f"Fallback path detection failed: {e}")

        raise RuntimeError(
            f"Cannot determine project root. Searched from {current_path} upward. "
            "Ensure the script is run from within the project directory."
        )

    def get_src_path(self) -> Path:
        """Get absolute path to src directory."""
        if self._src_path is None:
            self._src_path = self.detect_project_root() / "src"
        return self._src_path

    def get_ml_path(self) -> Path:
        """Get absolute path to machine_learning directory."""
        if self._ml_path is None:
            self._ml_path = self.get_src_path() / "machine_learning"
        return self._ml_path

    def get_pipeline_script_path(self) -> Path:
        """Get absolute path to pipeline_runner.py."""
        return self.get_ml_path() / "pipeline" / "pipeline_runner.py"

    def get_data_directories(self) -> Dict[str, Path]:
        """
        Get absolute paths to data directories.

        Returns:
            Dictionary with 'bld' and 'data' directory paths
        """
        root = self.detect_project_root()
        return {
            "bld": root / "bld",
            "data": root / "data",
            "logs": root / "logs",  # For production logs
        }

    def ensure_data_directories_exist(self) -> None:
        """Create data directories if they don't exist."""
        directories = self.get_data_directories()

        for dir_name, dir_path in directories.items():
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Ensured directory exists: {dir_path}")
            except PermissionError:
                raise PermissionError(
                    f"Cannot create {dir_name} directory at {dir_path}. "
                    "Check file permissions."
                )
            except Exception as e:
                raise RuntimeError(f"Failed to create {dir_name} directory: {e}")


# Global resolver instance
_resolver = ProjectPathResolver()


def resolve_project_paths() -> Dict[str, Path]:
    """
    Resolve all project paths for production deployment.

    Returns:
        Dictionary containing resolved absolute paths:
        - 'project_root': Project root directory
        - 'src': Source code directory
        - 'ml': Machine learning code directory
        - 'pipeline_script': Path to pipeline_runner.py
        - 'bld': Build/models directory
        - 'data': Data directory
        - 'logs': Logs directory
    """
    try:
        paths = {
            "project_root": _resolver.detect_project_root(),
            "src": _resolver.get_src_path(),
            "ml": _resolver.get_ml_path(),
            "pipeline_script": _resolver.get_pipeline_script_path(),
        }

        # Add data directories
        paths.update(_resolver.get_data_directories())

        logger.info("Successfully resolved all project paths")
        return paths

    except Exception as e:
        logger.error(f"Failed to resolve project paths: {e}")
        raise


def setup_working_directory() -> Tuple[Path, Path]:
    """
    Setup working directory for production execution.

    Changes working directory to project root and adds src to Python path.

    Returns:
        Tuple of (original_cwd, new_cwd)

    Raises:
        RuntimeError: If directory setup fails
    """
    original_cwd = Path.cwd()

    try:
        # Resolve paths
        paths = resolve_project_paths()
        project_root = paths["project_root"]
        src_path = paths["src"]
        ml_path = paths["ml"]

        # Change to project root
        os.chdir(project_root)
        new_cwd = Path.cwd()

        # Add necessary paths to Python path
        paths_to_add = [str(src_path), str(ml_path)]

        for path_str in paths_to_add:
            if path_str not in sys.path:
                sys.path.insert(0, path_str)
                logger.debug(f"Added to Python path: {path_str}")

        # Ensure data directories exist
        _resolver.ensure_data_directories_exist()

        logger.info(
            f"Working directory setup complete. Changed from {original_cwd} to {new_cwd}"
        )
        return original_cwd, new_cwd

    except Exception as e:
        # Restore original working directory on failure
        os.chdir(original_cwd)
        logger.error(f"Failed to setup working directory: {e}")
        raise RuntimeError(f"Working directory setup failed: {e}")


def validate_script_paths() -> bool:
    """
    Validate that all required script paths exist.

    Returns:
        True if all paths are valid, False otherwise
    """
    try:
        paths = resolve_project_paths()

        # Check critical files exist
        critical_files = [
            paths["pipeline_script"],
            paths["ml"] / "config.py",
        ]

        for file_path in critical_files:
            if not file_path.exists():
                logger.error(f"Critical file missing: {file_path}")
                return False

        # Check critical directories exist and are writable
        critical_dirs = [
            paths["bld"],
            paths["data"],
            paths["logs"],
        ]

        for dir_path in critical_dirs:
            if not dir_path.exists():
                logger.error(f"Critical directory missing: {dir_path}")
                return False

            # Test write permission
            test_file = dir_path / ".write_test"
            try:
                test_file.touch()
                test_file.unlink()
            except PermissionError:
                logger.error(f"No write permission to: {dir_path}")
                return False
            except Exception as e:
                logger.error(f"Cannot write to {dir_path}: {e}")
                return False

        logger.info("All script paths validated successfully")
        return True

    except Exception as e:
        logger.error(f"Path validation failed: {e}")
        return False


def get_absolute_config_path() -> Path:
    """
    Get absolute path to config.py for consistent configuration access.

    Returns:
        Absolute path to config.py
    """
    return _resolver.get_ml_path() / "config.py"
