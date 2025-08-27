"""
Test version of the production deployment wrapper for continuous learning pipeline.

This script mirrors deploy_pipeline.py exactly but with testing limitations:
- Calls test_pipeline_runner.py instead of pipeline_runner.py for limited MLB processing
- Limited to a small number of MLBs for testing purposes to avoid large file size issues
- All deployment wrapper functionality remains identical (logging, validation, cleanup)

Usage:
    python test_deploy_pipeline.py --mode daily
    python test_deploy_pipeline.py --mode full
    python test_deploy_pipeline.py --mode monthly
    python test_deploy_pipeline.py --since-date 2024-01-15
"""

import os
import sys
import time
import logging
import subprocess
from pathlib import Path
from typing import List, Optional

# Add deployment modules to path
sys.path.insert(0, str(Path(__file__).parent))

from deployment import (
    setup_production_logging,
    validate_production_environment,
    deployment_context,
    setup_working_directory,
    resolve_project_paths,
    emergency_cleanup,
    log_deployment_info,
)

# Testing constants
MAX_MLBS_FOR_TESTING = 5


class TestDeploymentPipelineRunner:
    """Test deployment pipeline runner that orchestrates testing deployment with limited MLBs."""

    def __init__(self):
        self.logger: Optional[logging.Logger] = None
        self.deployment_start_time: float = 0
        self.original_cwd: Optional[Path] = None
        self.pipeline_mode: str = "unknown"
        self.pipeline_args: List[str] = []

    def setup_deployment_environment(
        self, pipeline_mode: str, pipeline_args: List[str]
    ) -> bool:
        """
        Setup test deployment environment.

        Args:
            pipeline_mode: Pipeline execution mode (daily, full, monthly, since-date)
            pipeline_args: Arguments to pass to test_pipeline_runner.py

        Returns:
            True if setup successful, False otherwise
        """
        self.deployment_start_time = time.time()
        self.pipeline_mode = pipeline_mode
        self.pipeline_args = pipeline_args

        try:
            # Setup working directory and paths
            self.logger.info("Setting up TEST deployment environment...")
            self.original_cwd, new_cwd = setup_working_directory()

            # Log test deployment context
            deployment_context_info = {
                "pipeline_mode": pipeline_mode,
                "original_cwd": str(self.original_cwd),
                "working_directory": str(new_cwd),
                "python_executable": sys.executable,
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                "max_mlbs_for_testing": MAX_MLBS_FOR_TESTING,
                "environment_variables": {
                    "CONDA_DEFAULT_ENV": os.getenv("CONDA_DEFAULT_ENV", "None"),
                    "VIRTUAL_ENV": os.getenv("VIRTUAL_ENV", "None"),
                    "DB_HOST": os.getenv("DB_HOST", "default"),
                    "DB_USER": os.getenv("DB_USER", "default"),
                    "DB_NAME": os.getenv("DB_NAME", "default"),
                },
            }

            log_deployment_info(self.logger, **deployment_context_info)

            # Add test-specific logging
            self.logger.info(
                f"TEST MODE: Limited to {MAX_MLBS_FOR_TESTING} MLBs for testing purposes"
            )

            # Validate production environment
            self.logger.info("Validating test environment...")
            if not validate_production_environment():
                self.logger.error("Environment validation failed - cannot proceed")
                return False

            self.logger.info("Test deployment environment setup complete")
            return True

        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to setup test deployment environment: {e}")
            else:
                print(f"CRITICAL: Failed to setup test deployment environment: {e}")
            return False

    def execute_pipeline(self) -> subprocess.CompletedProcess:
        """
        Execute the TEST ML pipeline using subprocess.

        Returns:
            CompletedProcess result from test pipeline execution
        """
        try:
            # Get path to TEST pipeline script
            paths = resolve_project_paths()
            test_pipeline_script = (
                paths["pipeline_script"].parent / "test_pipeline_runner.py"
            )

            if not test_pipeline_script.exists():
                raise FileNotFoundError(
                    f"Test pipeline script not found: {test_pipeline_script}"
                )

            # Build command for test pipeline
            cmd = [sys.executable, str(test_pipeline_script)] + self.pipeline_args

            self.logger.info(f"Executing TEST pipeline command: {' '.join(cmd)}")
            self.logger.info(
                f"TEST LIMITATION: Processing maximum {MAX_MLBS_FOR_TESTING} MLBs"
            )

            # Execute test pipeline with proper environment
            env = os.environ.copy()
            env["PYTHONPATH"] = os.pathsep.join(
                [str(paths["src"]), str(paths["ml"]), env.get("PYTHONPATH", "")]
            )

            # Run the test pipeline
            result = subprocess.run(
                cmd,
                cwd=paths["project_root"],
                env=env,
                capture_output=False,  # Let output go to console/logs
                text=True,
                check=False,  # Don't raise exception on non-zero exit
            )

            return result

        except Exception as e:
            self.logger.error(f"Test pipeline execution failed: {e}")
            # Create a mock result for consistent error handling
            return subprocess.CompletedProcess(
                args=cmd if "cmd" in locals() else [],
                returncode=1,
                stdout="",
                stderr=str(e),
            )

    def handle_pipeline_result(self, result: subprocess.CompletedProcess) -> int:
        """
        Handle test pipeline execution result and determine exit code.

        Args:
            result: CompletedProcess result from test pipeline execution

        Returns:
            Exit code for test deployment process
        """
        execution_time = time.time() - self.deployment_start_time

        if result.returncode == 0:
            self.logger.info("=" * 60)
            self.logger.info("TEST DEPLOYMENT SUCCESSFUL")
            self.logger.info(f"Pipeline mode: {self.pipeline_mode}")
            self.logger.info(f"MLB limit: {MAX_MLBS_FOR_TESTING} MLBs (TEST MODE)")
            self.logger.info(f"Execution time: {execution_time:.2f} seconds")
            self.logger.info("=" * 60)
            return 0
        else:
            self.logger.error("=" * 60)
            self.logger.error("TEST DEPLOYMENT FAILED")
            self.logger.error(f"Pipeline mode: {self.pipeline_mode}")
            self.logger.error(f"Exit code: {result.returncode}")
            self.logger.error(f"Execution time: {execution_time:.2f} seconds")

            if result.stderr:
                self.logger.error(f"Error output: {result.stderr}")

            # Log troubleshooting suggestions
            self.logger.error("Test troubleshooting suggestions:")
            self.logger.error("1. Check database connectivity (DB_PASSWORD set?)")
            self.logger.error("2. Verify file permissions in data directories")
            self.logger.error("3. Check available memory and disk space")
            self.logger.error("4. Review test pipeline logs for specific error details")
            self.logger.error(
                "5. Verify test_pipeline_runner.py exists and is functional"
            )

            self.logger.error("=" * 60)
            return result.returncode

    def run_deployment(self, pipeline_mode: str, pipeline_args: List[str]) -> int:
        """
        Main test deployment execution method.

        Args:
            pipeline_mode: Pipeline execution mode
            pipeline_args: Arguments to pass to test pipeline

        Returns:
            Exit code (0 = success, non-zero = failure)
        """
        try:
            # Setup test deployment environment
            if not self.setup_deployment_environment(pipeline_mode, pipeline_args):
                return 1

            # Execute test pipeline
            result = self.execute_pipeline()

            # Handle result and determine exit code
            return self.handle_pipeline_result(result)

        except KeyboardInterrupt:
            self.logger.warning("Test deployment interrupted by user (Ctrl+C)")
            return 130  # Standard exit code for SIGINT

        except Exception as e:
            if self.logger:
                self.logger.critical(f"Unexpected test deployment error: {e}")
            else:
                print(f"CRITICAL: Unexpected test deployment error: {e}")

            # Try emergency cleanup
            try:
                emergency_cleanup()
            except Exception as cleanup_error:
                if self.logger:
                    self.logger.error(f"Emergency cleanup failed: {cleanup_error}")
                else:
                    print(f"Emergency cleanup failed: {cleanup_error}")

            return 1


def parse_arguments() -> tuple[str, List[str]]:
    """
    Parse command line arguments and determine pipeline mode.

    Returns:
        Tuple of (pipeline_mode, pipeline_args)
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Test deployment wrapper for continuous learning pipeline (LIMITED MLBs)",
        epilog=f"All arguments are passed through to test_pipeline_runner.py. Limited to {MAX_MLBS_FOR_TESTING} MLBs for testing.",
    )

    # Add same arguments as pipeline_runner.py for compatibility
    parser.add_argument(
        "--mode",
        choices=["daily", "full", "monthly"],
        default="daily",
        help="Pipeline execution mode (default: daily)",
    )
    parser.add_argument(
        "--since-date",
        type=str,
        help="Run incremental updates since specific date (YYYY-MM-DD format)",
    )
    parser.add_argument(
        "since_date_positional",
        nargs="?",
        help="Alternative way to specify since-date for backward compatibility",
    )

    # Parse known args to allow for future pipeline extensions
    args, unknown_args = parser.parse_known_args()

    # Determine pipeline mode
    since_date = args.since_date or args.since_date_positional
    if since_date:
        pipeline_mode = "since-date"
    else:
        pipeline_mode = args.mode

    # Build args list for test_pipeline_runner.py
    pipeline_args = []

    if since_date:
        if args.since_date:
            pipeline_args.extend(["--since-date", since_date])
        else:
            pipeline_args.append(since_date)  # Positional argument
    else:
        pipeline_args.extend(["--mode", args.mode])

    # Add any unknown arguments
    pipeline_args.extend(unknown_args)

    return pipeline_mode, pipeline_args


def main() -> int:
    """
    Main entry point for test deployment wrapper.

    Returns:
        Exit code (0 = success, non-zero = failure)
    """
    runner = None

    try:
        # Parse command line arguments
        pipeline_mode, pipeline_args = parse_arguments()

        # Initialize test deployment runner
        runner = TestDeploymentPipelineRunner()

        # Setup production logging with test identifier
        runner.logger = setup_production_logging(
            mode="test", pipeline_mode=f"test_{pipeline_mode}"
        )

        # Log test mode startup
        runner.logger.info("=" * 60)
        runner.logger.info("STARTING TEST DEPLOYMENT WRAPPER")
        runner.logger.info(f"Test mode: {pipeline_mode}")
        runner.logger.info(f"MLB limit: {MAX_MLBS_FOR_TESTING} MLBs")
        runner.logger.info("Target script: test_pipeline_runner.py")
        runner.logger.info("=" * 60)

        # Use deployment context for automatic cleanup
        with deployment_context(f"test_deploy_pipeline_{pipeline_mode}"):
            return runner.run_deployment(pipeline_mode, pipeline_args)

    except Exception as e:
        # Critical error before logging is setup
        print(f"CRITICAL: Test deployment initialization failed: {e}")

        # Try emergency cleanup
        try:
            emergency_cleanup()
        except Exception:
            pass  # Can't do much more

        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
