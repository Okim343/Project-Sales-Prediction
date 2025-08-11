"""
Model storage utilities for continuous learning pipeline.
Handles saving, loading, and archiving of trained XGBoost models.
"""

import logging
import pickle
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


def save_models(models: Dict[str, Any], filepath: Path) -> None:
    """
    Save a dictionary of trained models to disk using pickle.

    Args:
        models (Dict[str, Any]): Dictionary of models keyed by MLB.
        filepath (Path): The file path where the models should be saved.

    Raises:
        TypeError: If models is not a dictionary or filepath is not a Path object.
        OSError: If file write operation fails.
    """
    _fail_if_invalid_models_dict(models)
    _fail_if_invalid_filepath(filepath)

    try:
        # Ensure parent directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with filepath.open("wb") as f:
            pickle.dump(models, f)

        logger.info(f"Successfully saved {len(models)} models to {filepath}")

    except Exception as e:
        error_msg = f"Failed to save models to {filepath}: {e}"
        logger.error(error_msg)
        raise OSError(error_msg) from e


def load_models(filepath: Path) -> Dict[str, Any]:
    """
    Load models from disk using pickle.

    Args:
        filepath (Path): The file path from which to load models.

    Returns:
        Dict[str, Any]: Dictionary of models keyed by MLB.

    Raises:
        FileNotFoundError: If the model file does not exist.
        OSError: If file read operation fails.
        TypeError: If filepath is not a Path object.
    """
    _fail_if_invalid_filepath(filepath)

    if not filepath.exists():
        error_msg = f"Model file does not exist: {filepath}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    try:
        with filepath.open("rb") as f:
            models = pickle.load(f)

        logger.info(f"Successfully loaded {len(models)} models from {filepath}")
        return models

    except Exception as e:
        error_msg = f"Failed to load models from {filepath}: {e}"
        logger.error(error_msg)
        raise OSError(error_msg) from e


def archive_models(current_path: Path, archive_dir: Path) -> None:
    """
    Archive existing models by moving them to an archive directory with timestamp.

    Args:
        current_path (Path): Path to the current model file to archive.
        archive_dir (Path): Directory where archived models should be stored.

    Raises:
        TypeError: If paths are not Path objects.
        OSError: If file operations fail.
    """
    _fail_if_invalid_filepath(current_path)
    _fail_if_invalid_filepath(archive_dir)

    if not current_path.exists():
        logger.debug(f"No existing model file to archive at {current_path}")
        return

    try:
        # Ensure archive directory exists
        archive_dir.mkdir(parents=True, exist_ok=True)

        # Create timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archived_filename = f"{current_path.stem}_{timestamp}{current_path.suffix}"
        archived_path = archive_dir / archived_filename

        # Move the file to archive
        shutil.move(str(current_path), str(archived_path))

        logger.info(f"Archived existing models from {current_path} to {archived_path}")

    except Exception as e:
        error_msg = (
            f"Failed to archive models from {current_path} to {archive_dir}: {e}"
        )
        logger.error(error_msg)
        raise OSError(error_msg) from e


def _fail_if_invalid_models_dict(models: Any) -> None:
    """Raise an error if models is not a dictionary."""
    if not isinstance(models, dict):
        error_msg = f"'models' must be a dictionary, got {type(models)}."
        raise TypeError(error_msg)


def _fail_if_invalid_filepath(filepath: Any) -> None:
    """Raise an error if filepath is not a Path object."""
    if not isinstance(filepath, Path):
        error_msg = f"'filepath' must be a Path object, got {type(filepath)}."
        raise TypeError(error_msg)
