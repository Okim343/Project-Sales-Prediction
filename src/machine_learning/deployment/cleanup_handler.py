"""
Cleanup handlers for graceful shutdown and resource management.

Provides signal handlers, resource cleanup, and process lock management
to ensure production deployments handle interruptions gracefully.
"""

import os
import sys
import signal
import time
import logging
import threading
from pathlib import Path
from typing import Optional, Callable, Dict, Any, List
from contextlib import contextmanager

from .path_resolver import resolve_project_paths


logger = logging.getLogger(__name__)


class ProcessLock:
    """Process lock to prevent concurrent pipeline executions."""

    def __init__(self, lock_name: str = "deploy_pipeline"):
        self.lock_name = lock_name
        self.lock_file: Optional[Path] = None
        self.lock_acquired = False

    def _get_lock_file_path(self) -> Path:
        """Get path to lock file."""
        try:
            paths = resolve_project_paths()
            lock_dir = paths.get("logs", paths["project_root"] / "logs")
        except Exception:
            lock_dir = Path.cwd()

        return lock_dir / f"{self.lock_name}.lock"

    def acquire(self, timeout: float = 60.0) -> bool:
        """
        Acquire process lock.

        Args:
            timeout: Maximum time to wait for lock in seconds

        Returns:
            True if lock acquired, False otherwise
        """
        if self.lock_acquired:
            return True

        self.lock_file = self._get_lock_file_path()
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                if not self.lock_file.exists():
                    # Create lock file with current process info
                    lock_info = {
                        "pid": os.getpid(),
                        "start_time": time.time(),
                        "command": " ".join(sys.argv),
                    }

                    # Use atomic write operation
                    temp_file = self.lock_file.with_suffix(".tmp")
                    with open(temp_file, "w") as f:
                        import json

                        json.dump(lock_info, f)

                    temp_file.rename(self.lock_file)
                    self.lock_acquired = True

                    logger.info(f"Acquired process lock: {self.lock_file}")
                    return True
                else:
                    # Check if existing lock is stale
                    try:
                        with open(self.lock_file, "r") as f:
                            import json

                            existing_lock = json.load(f)

                        existing_pid = existing_lock.get("pid")
                        if existing_pid and not self._is_process_running(existing_pid):
                            # Stale lock, remove it
                            self.lock_file.unlink()
                            logger.warning(
                                f"Removed stale lock file from PID {existing_pid}"
                            )
                            continue
                        else:
                            # Active process, wait
                            logger.info(
                                f"Waiting for process lock (held by PID {existing_pid})..."
                            )
                            time.sleep(5)

                    except (json.JSONDecodeError, KeyError, OSError) as e:
                        # Corrupted lock file, remove it
                        logger.warning(f"Removing corrupted lock file: {e}")
                        try:
                            self.lock_file.unlink()
                        except OSError:
                            pass
                        continue

            except Exception as e:
                logger.error(f"Error acquiring process lock: {e}")
                time.sleep(1)

        logger.error(f"Failed to acquire process lock within {timeout} seconds")
        return False

    def release(self) -> None:
        """Release process lock."""
        if not self.lock_acquired or not self.lock_file:
            return

        try:
            if self.lock_file.exists():
                self.lock_file.unlink()
                logger.info(f"Released process lock: {self.lock_file}")
        except Exception as e:
            logger.error(f"Error releasing process lock: {e}")
        finally:
            self.lock_acquired = False
            self.lock_file = None

    def _is_process_running(self, pid: int) -> bool:
        """Check if a process is still running."""
        try:
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False

    def __enter__(self):
        if not self.acquire():
            raise RuntimeError("Could not acquire process lock")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


class ResourceManager:
    """Manages cleanup of resources during shutdown."""

    def __init__(self):
        self.cleanup_functions: List[Callable[[], None]] = []
        self.resources: Dict[str, Any] = {}
        self._shutdown_initiated = False
        self._lock = threading.Lock()

    def register_cleanup(
        self, cleanup_func: Callable[[], None], name: str = None
    ) -> None:
        """
        Register a cleanup function to be called during shutdown.

        Args:
            cleanup_func: Function to call during cleanup
            name: Optional name for the cleanup function
        """
        with self._lock:
            self.cleanup_functions.append(cleanup_func)
            if name:
                logger.debug(f"Registered cleanup function: {name}")

    def register_resource(
        self, name: str, resource: Any, cleanup_method: str = "close"
    ) -> None:
        """
        Register a resource that needs cleanup.

        Args:
            name: Name of the resource
            resource: The resource object
            cleanup_method: Method name to call for cleanup (default: 'close')
        """
        with self._lock:
            self.resources[name] = (resource, cleanup_method)
            logger.debug(f"Registered resource for cleanup: {name}")

    def cleanup_resources(self) -> None:
        """Clean up all registered resources."""
        if self._shutdown_initiated:
            return

        with self._lock:
            self._shutdown_initiated = True

            logger.info("Starting resource cleanup...")

            # Clean up registered resources
            for name, (resource, cleanup_method) in self.resources.items():
                try:
                    if hasattr(resource, cleanup_method):
                        method = getattr(resource, cleanup_method)
                        method()
                        logger.debug(f"Cleaned up resource: {name}")
                    else:
                        logger.warning(
                            f"Resource {name} has no method {cleanup_method}"
                        )
                except Exception as e:
                    logger.error(f"Error cleaning up resource {name}: {e}")

            # Call cleanup functions
            for i, cleanup_func in enumerate(self.cleanup_functions):
                try:
                    cleanup_func()
                    logger.debug(f"Executed cleanup function {i}")
                except Exception as e:
                    logger.error(f"Error in cleanup function {i}: {e}")

            logger.info("Resource cleanup complete")


class DeploymentCleanupHandler:
    """Main cleanup handler for deployment processes."""

    def __init__(self):
        self.resource_manager = ResourceManager()
        self.process_lock: Optional[ProcessLock] = None
        self._signal_handlers_registered = False
        self._original_handlers: Dict[int, Any] = {}
        self._cleanup_in_progress = False

    def register_signal_handlers(self) -> None:
        """Register signal handlers for graceful shutdown."""
        if self._signal_handlers_registered:
            return

        # Store original handlers
        signals_to_handle = [signal.SIGTERM, signal.SIGINT]

        # Add SIGHUP on Unix systems
        if hasattr(signal, "SIGHUP"):
            signals_to_handle.append(signal.SIGHUP)

        for sig in signals_to_handle:
            try:
                original_handler = signal.signal(sig, self._signal_handler)
                self._original_handlers[sig] = original_handler
                logger.debug(f"Registered signal handler for {sig.name}")
            except (OSError, ValueError) as e:
                # Some signals might not be available on all platforms
                logger.warning(f"Could not register handler for {sig.name}: {e}")

        self._signal_handlers_registered = True
        logger.info("Signal handlers registered for graceful shutdown")

    def _signal_handler(self, signum: int, frame) -> None:
        """Handle shutdown signals."""
        signal_name = (
            signal.Signals(signum).name if hasattr(signal, "Signals") else str(signum)
        )
        logger.info(f"Received signal {signal_name}, initiating graceful shutdown...")

        self.cleanup_and_exit(exit_code=128 + signum)

    def register_process_lock(self, lock_name: str = "deploy_pipeline") -> ProcessLock:
        """
        Register a process lock that will be cleaned up on exit.

        Args:
            lock_name: Name for the lock file

        Returns:
            ProcessLock instance
        """
        self.process_lock = ProcessLock(lock_name)

        # Register lock cleanup
        self.resource_manager.register_cleanup(
            self.process_lock.release, name="process_lock"
        )

        return self.process_lock

    def register_database_cleanup(self) -> None:
        """Register database connection cleanup."""
        try:
            # Import database manager if available
            sys.path.insert(0, str(resolve_project_paths()["ml"]))
            from database_utils import db_manager

            def cleanup_database():
                try:
                    db_manager.close_connection()
                    logger.debug("Database connections closed")
                except Exception as e:
                    logger.error(f"Error closing database connections: {e}")

            self.resource_manager.register_cleanup(
                cleanup_database, name="database_connections"
            )

        except Exception as e:
            logger.warning(f"Could not register database cleanup: {e}")

    def register_logging_cleanup(self) -> None:
        """Register logging cleanup."""
        from .logging_config import cleanup_logging

        self.resource_manager.register_cleanup(cleanup_logging, name="logging_handlers")

    def register_temp_file_cleanup(self, temp_dir: Optional[Path] = None) -> None:
        """
        Register cleanup of temporary files.

        Args:
            temp_dir: Directory to clean up (optional)
        """

        def cleanup_temp_files():
            try:
                if temp_dir and temp_dir.exists():
                    # Remove temporary files
                    for temp_file in temp_dir.glob("*.tmp"):
                        try:
                            temp_file.unlink()
                            logger.debug(f"Removed temp file: {temp_file}")
                        except Exception as e:
                            logger.error(f"Error removing temp file {temp_file}: {e}")

                # Clean up any .write_test files
                try:
                    paths = resolve_project_paths()
                    for dir_path in [paths["bld"], paths["data"], paths["logs"]]:
                        for test_file in dir_path.glob(".write_test*"):
                            try:
                                test_file.unlink()
                                logger.debug(f"Removed test file: {test_file}")
                            except Exception:
                                pass
                except Exception:
                    pass  # Best effort cleanup

            except Exception as e:
                logger.error(f"Error cleaning up temporary files: {e}")

        self.resource_manager.register_cleanup(
            cleanup_temp_files, name="temporary_files"
        )

    def cleanup_and_exit(self, exit_code: int = 0) -> None:
        """
        Perform cleanup and exit.

        Args:
            exit_code: Exit code to use
        """
        if self._cleanup_in_progress:
            logger.warning("Cleanup already in progress, forcing exit")
            os._exit(exit_code)

        self._cleanup_in_progress = True

        try:
            logger.info("Initiating deployment cleanup...")

            # Clean up resources
            self.resource_manager.cleanup_resources()

            # Restore original signal handlers
            for sig, original_handler in self._original_handlers.items():
                try:
                    signal.signal(sig, original_handler)
                except Exception as e:
                    logger.debug(f"Error restoring signal handler for {sig}: {e}")

            logger.info("Deployment cleanup complete")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            exit_code = max(exit_code, 1)  # Ensure non-zero exit code

        finally:
            sys.exit(exit_code)


# Global cleanup handler instance
_cleanup_handler: Optional[DeploymentCleanupHandler] = None


def register_cleanup_handlers(
    lock_name: str = "deploy_pipeline",
) -> DeploymentCleanupHandler:
    """
    Register cleanup handlers for production deployment.

    Args:
        lock_name: Name for the process lock

    Returns:
        DeploymentCleanupHandler instance
    """
    global _cleanup_handler

    if _cleanup_handler is None:
        _cleanup_handler = DeploymentCleanupHandler()

        # Register signal handlers
        _cleanup_handler.register_signal_handlers()

        # Register process lock
        _cleanup_handler.register_process_lock(lock_name)

        # Register common cleanup tasks
        _cleanup_handler.register_database_cleanup()
        _cleanup_handler.register_logging_cleanup()
        _cleanup_handler.register_temp_file_cleanup()

        logger.info("Cleanup handlers registered for production deployment")

    return _cleanup_handler


def cleanup_resources() -> None:
    """Manually trigger resource cleanup."""
    global _cleanup_handler

    if _cleanup_handler:
        _cleanup_handler.resource_manager.cleanup_resources()


@contextmanager
def deployment_context(lock_name: str = "deploy_pipeline"):
    """
    Context manager for deployment with automatic cleanup.

    Args:
        lock_name: Name for the process lock

    Usage:
        with deployment_context("daily_pipeline") as handler:
            # Run deployment logic
            pass
    """
    handler = register_cleanup_handlers(lock_name)

    # Acquire process lock
    if handler.process_lock and not handler.process_lock.acquire():
        raise RuntimeError(
            "Could not acquire process lock - another instance may be running"
        )

    try:
        yield handler
    except KeyboardInterrupt:
        logger.info("Deployment interrupted by user")
        raise
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        raise
    finally:
        # Cleanup is automatically handled by signal handlers
        # But we can do explicit cleanup here if needed
        pass


def emergency_cleanup() -> None:
    """Emergency cleanup function for critical failures."""
    try:
        logger.critical("Performing emergency cleanup...")

        # Try to release any locks
        try:
            paths = resolve_project_paths()
            lock_dir = paths.get("logs", paths["project_root"] / "logs")
            for lock_file in lock_dir.glob("*.lock"):
                try:
                    with open(lock_file, "r") as f:
                        import json

                        lock_info = json.load(f)

                    if lock_info.get("pid") == os.getpid():
                        lock_file.unlink()
                        logger.info(f"Removed own lock file: {lock_file}")
                except Exception:
                    pass  # Best effort
        except Exception:
            pass  # Best effort

        # Try to close database connections
        try:
            sys.path.insert(0, str(resolve_project_paths()["ml"]))
            from database_utils import db_manager

            db_manager.close_connection()
        except Exception:
            pass  # Best effort

        logger.info("Emergency cleanup complete")

    except Exception as e:
        # Can't do much more at this point
        print(f"Emergency cleanup failed: {e}")
