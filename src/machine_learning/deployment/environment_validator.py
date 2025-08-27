"""
Environment validation utilities for production deployment.

Validates the production environment before pipeline execution to catch
issues early and provide clear error messages.
"""

import os
import sys
import logging
import importlib
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from sqlalchemy import text

from .path_resolver import resolve_project_paths, validate_script_paths


logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of an environment validation check."""

    passed: bool
    message: str
    details: Optional[Dict[str, Any]] = None

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"[{status}] {self.message}"


class EnvironmentValidator:
    """Comprehensive environment validation for production deployment."""

    def __init__(self):
        self.results: List[ValidationResult] = []
        self.critical_failures: List[ValidationResult] = []

        # Load environment variables from .env file if available
        # This must happen BEFORE validation checks to ensure env vars are loaded
        try:
            from dotenv import load_dotenv

            # Load .env file from project root
            env_file = resolve_project_paths()["project_root"] / ".env"
            if env_file.exists():
                load_dotenv(env_file)
                logger.debug(f"Loaded environment variables from {env_file}")
            else:
                load_dotenv()  # Try loading from current directory
                logger.debug("Attempted to load .env from current directory")
        except ImportError:
            # python-dotenv not installed, continue without it
            logger.debug("python-dotenv not available, skipping .env file loading")
            pass
        except Exception as e:
            # Log but don't fail on .env loading issues
            logger.debug(f"Could not load .env file: {e}")
            pass

    def _add_result(self, result: ValidationResult, critical: bool = False) -> None:
        """Add validation result and track critical failures."""
        self.results.append(result)
        if not result.passed and critical:
            self.critical_failures.append(result)

        # Log the result
        if result.passed:
            logger.info(f"✓ {result.message}")
        else:
            log_func = logger.error if critical else logger.warning
            log_func(f"✗ {result.message}")
            if result.details:
                for key, value in result.details.items():
                    log_func(f"  {key}: {value}")

    def validate_python_environment(self) -> None:
        """Validate Python version and core environment."""
        # Check Python version
        python_version = sys.version_info
        if python_version >= (3, 11):
            self._add_result(
                ValidationResult(
                    passed=True,
                    message=f"Python version {python_version.major}.{python_version.minor} is supported",
                    details={
                        "version": f"{python_version.major}.{python_version.minor}.{python_version.micro}"
                    },
                )
            )
        else:
            self._add_result(
                ValidationResult(
                    passed=False,
                    message=f"Python version {python_version.major}.{python_version.minor} may be too old (recommend 3.11+)",
                    details={
                        "current": f"{python_version.major}.{python_version.minor}.{python_version.micro}"
                    },
                ),
                critical=False,
            )  # Warning, not critical

        # Check if we're in a conda/virtual environment
        in_conda = "CONDA_DEFAULT_ENV" in os.environ
        in_venv = hasattr(sys, "real_prefix") or (
            hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
        )

        if in_conda or in_venv:
            env_name = os.environ.get("CONDA_DEFAULT_ENV", "virtual environment")
            self._add_result(
                ValidationResult(
                    passed=True,
                    message=f"Running in isolated environment: {env_name}",
                    details={"conda": in_conda, "venv": in_venv},
                )
            )
        else:
            self._add_result(
                ValidationResult(
                    passed=False,
                    message="Not running in conda or virtual environment - may cause dependency conflicts",
                    details={"recommendation": "Use conda activate fcast_project"},
                ),
                critical=False,
            )

    def validate_required_packages(self) -> None:
        """Validate that required Python packages are available."""
        required_packages = [
            ("pandas", "2.0.0"),
            ("numpy", "1.20.0"),
            ("scikit-learn", "1.0.0"),
            ("xgboost", "1.0.0"),
            ("sqlalchemy", "1.4.0"),
            ("psycopg2", "2.8.0"),  # May be psycopg2-binary
            ("plotly", "5.0.0"),
            ("dash", "2.0.0"),
        ]

        # Special cases where PyPI name differs from import name
        import_name_mapping = {
            "scikit-learn": "sklearn",
        }

        for package_name, min_version in required_packages:
            try:
                # Try importing the package
                if package_name == "psycopg2":
                    # Special case for psycopg2/psycopg2-binary
                    try:
                        import psycopg2

                        version = psycopg2.__version__
                    except ImportError:
                        import psycopg2_binary as psycopg2

                        version = psycopg2.__version__
                else:
                    # Determine the correct import name
                    import_name = import_name_mapping.get(
                        package_name, package_name.replace("-", "_")
                    )
                    module = importlib.import_module(import_name)
                    version = getattr(module, "__version__", "unknown")

                self._add_result(
                    ValidationResult(
                        passed=True,
                        message=f"Package {package_name} is available",
                        details={"version": version, "required": f">={min_version}"},
                    )
                )

            except ImportError as e:
                self._add_result(
                    ValidationResult(
                        passed=False,
                        message=f"Required package {package_name} is not available",
                        details={
                            "error": str(e),
                            "install_cmd": f"pip install {package_name}",
                        },
                    ),
                    critical=True,
                )

    def validate_database_connectivity(self) -> None:
        """Validate database connection and access."""
        try:
            # Import database utilities
            sys.path.insert(0, str(resolve_project_paths()["ml"]))
            from config import DatabaseConfig
            from database_utils import DatabaseManager

            # Test database connection
            db_config = DatabaseConfig()
            db_manager = DatabaseManager()

            # Try to connect and run a simple query
            with db_manager.engine.connect() as conn:
                result = conn.execute(text("SELECT 1 as test")).fetchone()
                if result and result[0] == 1:
                    self._add_result(
                        ValidationResult(
                            passed=True,
                            message="Database connection successful",
                            details={
                                "host": db_config.HOST,
                                "database": db_config.DBNAME,
                                "user": db_config.USER,
                            },
                        )
                    )
                else:
                    raise RuntimeError("Query test failed")

            # Test access to required tables/views
            with db_manager.engine.connect() as conn:
                # Check if source view exists
                view_check = conn.execute(
                    text(f"SELECT 1 FROM {db_config.VIEW} LIMIT 1")
                ).fetchone()
                if view_check:
                    self._add_result(
                        ValidationResult(
                            passed=True,
                            message=f"Source view {db_config.VIEW} is accessible",
                        )
                    )

            db_manager.close_connection()

        except Exception as e:
            self._add_result(
                ValidationResult(
                    passed=False,
                    message="Database connection failed",
                    details={
                        "error": str(e),
                        "check": "Verify DB_PASSWORD environment variable and network access",
                    },
                ),
                critical=True,
            )

    def validate_file_permissions(self) -> None:
        """Validate file system permissions for required directories."""
        try:
            paths = resolve_project_paths()

            # Directories that need write access
            write_directories = [
                ("bld", paths["bld"]),
                ("data", paths["data"]),
                ("logs", paths["logs"]),
            ]

            for dir_name, dir_path in write_directories:
                try:
                    # Test write permission
                    test_file = dir_path / f".write_test_{os.getpid()}"
                    test_file.write_text("test")
                    test_file.unlink()

                    self._add_result(
                        ValidationResult(
                            passed=True,
                            message=f"Write access to {dir_name} directory confirmed",
                            details={"path": str(dir_path)},
                        )
                    )

                except PermissionError:
                    self._add_result(
                        ValidationResult(
                            passed=False,
                            message=f"No write permission to {dir_name} directory",
                            details={
                                "path": str(dir_path),
                                "fix": f"chmod 755 {dir_path}",
                            },
                        ),
                        critical=True,
                    )

                except Exception as e:
                    self._add_result(
                        ValidationResult(
                            passed=False,
                            message=f"Cannot access {dir_name} directory",
                            details={"path": str(dir_path), "error": str(e)},
                        ),
                        critical=True,
                    )

            # Check that critical files exist
            if not validate_script_paths():
                self._add_result(
                    ValidationResult(
                        passed=False,
                        message="Critical script files are missing or inaccessible",
                        details={
                            "check": "Verify project structure and file permissions"
                        },
                    ),
                    critical=True,
                )
            else:
                self._add_result(
                    ValidationResult(
                        passed=True, message="All critical script files are accessible"
                    )
                )

        except Exception as e:
            self._add_result(
                ValidationResult(
                    passed=False,
                    message="File permission validation failed",
                    details={"error": str(e)},
                ),
                critical=True,
            )

    def validate_environment_variables(self) -> None:
        """Validate required environment variables."""
        # Critical environment variables
        critical_vars = [
            "DB_PASSWORD",
        ]

        # Optional but recommended environment variables
        optional_vars = [
            ("DB_HOST", "172.27.40.210"),
            ("DB_USER", "postgres"),
            ("DB_NAME", "Mercado Livre"),
            ("FORECAST_DAYS_LONG", "90"),
        ]

        # Check critical variables
        for var_name in critical_vars:
            if os.getenv(var_name):
                self._add_result(
                    ValidationResult(
                        passed=True,
                        message=f"Environment variable {var_name} is set",
                    )
                )
            else:
                self._add_result(
                    ValidationResult(
                        passed=False,
                        message=f"Critical environment variable {var_name} is not set",
                        details={"fix": f"export {var_name}=your_value"},
                    ),
                    critical=True,
                )

        # Check optional variables
        for var_name, default_value in optional_vars:
            current_value = os.getenv(var_name, default_value)
            self._add_result(
                ValidationResult(
                    passed=True,
                    message=f"Environment variable {var_name} = {current_value}",
                    details={
                        "default": default_value if not os.getenv(var_name) else None
                    },
                )
            )

    def validate_memory_resources(self) -> None:
        """Validate available system resources."""
        try:
            import psutil

            # Check available memory
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)

            if available_gb >= 4.0:
                self._add_result(
                    ValidationResult(
                        passed=True,
                        message=f"Sufficient memory available: {available_gb:.1f} GB",
                    )
                )
            else:
                self._add_result(
                    ValidationResult(
                        passed=False,
                        message=f"Low memory available: {available_gb:.1f} GB (recommend 4+ GB)",
                        details={
                            "available": f"{available_gb:.1f} GB",
                            "total": f"{memory.total/(1024**3):.1f} GB",
                        },
                    ),
                    critical=False,
                )  # Warning, not critical

            # Check disk space
            paths = resolve_project_paths()
            disk_usage = psutil.disk_usage(paths["project_root"])
            free_gb = disk_usage.free / (1024**3)

            if free_gb >= 10.0:
                self._add_result(
                    ValidationResult(
                        passed=True,
                        message=f"Sufficient disk space: {free_gb:.1f} GB free",
                    )
                )
            else:
                self._add_result(
                    ValidationResult(
                        passed=False,
                        message=f"Low disk space: {free_gb:.1f} GB free (recommend 10+ GB)",
                        details={"free": f"{free_gb:.1f} GB"},
                    ),
                    critical=False,
                )

        except ImportError:
            self._add_result(
                ValidationResult(
                    passed=False,
                    message="psutil not available - cannot check system resources",
                    details={"install": "pip install psutil"},
                ),
                critical=False,
            )
        except Exception as e:
            self._add_result(
                ValidationResult(
                    passed=False,
                    message="System resource validation failed",
                    details={"error": str(e)},
                ),
                critical=False,
            )

    def run_all_validations(self) -> bool:
        """
        Run all environment validations.

        Returns:
            True if all critical validations pass, False otherwise
        """
        logger.info("Starting production environment validation...")

        # Clear previous results
        self.results.clear()
        self.critical_failures.clear()

        # Run all validations
        self.validate_python_environment()
        self.validate_required_packages()
        self.validate_environment_variables()
        self.validate_file_permissions()
        self.validate_database_connectivity()
        self.validate_memory_resources()

        # Summary
        total_checks = len(self.results)
        passed_checks = sum(1 for r in self.results if r.passed)
        failed_checks = total_checks - passed_checks
        critical_failures_count = len(self.critical_failures)

        logger.info(
            f"Environment validation complete: {passed_checks}/{total_checks} checks passed"
        )

        if critical_failures_count > 0:
            logger.error(f"Critical validation failures: {critical_failures_count}")
            logger.error(
                "The following critical issues must be fixed before proceeding:"
            )
            for failure in self.critical_failures:
                logger.error(f"  - {failure.message}")
            return False

        if failed_checks > 0:
            logger.warning(f"Non-critical warnings: {failed_checks}")
            logger.warning("Consider addressing these warnings for optimal performance")

        return True

    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of validation results."""
        return {
            "total_checks": len(self.results),
            "passed": sum(1 for r in self.results if r.passed),
            "failed": sum(1 for r in self.results if not r.passed),
            "critical_failures": len(self.critical_failures),
            "can_proceed": len(self.critical_failures) == 0,
            "results": [
                {"message": r.message, "passed": r.passed, "details": r.details or {}}
                for r in self.results
            ],
        }


def validate_production_environment() -> bool:
    """
    Run comprehensive environment validation.

    Returns:
        True if environment is ready for production deployment, False otherwise
    """
    validator = EnvironmentValidator()
    return validator.run_all_validations()


def quick_environment_check() -> Tuple[bool, List[str]]:
    """
    Run a quick environment check for essential components.

    Returns:
        Tuple of (is_ready, issues_list)
    """
    issues = []

    # Check Python version
    if sys.version_info < (3, 10):
        issues.append(
            f"Python version {sys.version_info.major}.{sys.version_info.minor} may be too old"
        )

    # Check database password
    if not os.getenv("DB_PASSWORD"):
        issues.append("DB_PASSWORD environment variable not set")

    # Check core packages
    try:
        import pandas  # noqa: F401
        import numpy  # noqa: F401
        import xgboost  # noqa: F401
        import sqlalchemy  # noqa: F401
    except ImportError as e:
        issues.append(f"Required package missing: {e}")

    # Check basic file access
    try:
        paths = resolve_project_paths()
        test_file = paths["bld"] / ".test"
        test_file.touch()
        test_file.unlink()
    except Exception:
        issues.append("Cannot write to data directories")

    return len(issues) == 0, issues
