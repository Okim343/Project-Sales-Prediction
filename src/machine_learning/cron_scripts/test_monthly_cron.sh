#!/bin/bash

# Test Monthly Cron Script for Sales Forecasting Pipeline
# This script mirrors monthly_cron.sh exactly but calls test_deploy_pipeline.py for safe testing
# Limited to 5 MLBs for testing purposes - designed for production cron execution testing

set -euo pipefail  # Exit on any error, undefined variables, or pipe failures

# =============================================================================
# Configuration
# =============================================================================

# Environment and script configuration
CONDA_ENV_NAME="fcast_project"
PIPELINE_MODE="monthly"
SCRIPT_NAME="test_monthly_cron.sh"

# Exit codes
EXIT_SUCCESS=0
EXIT_CONDA_ERROR=10
EXIT_ACTIVATION_ERROR=11
EXIT_PROJECT_ERROR=12
EXIT_PIPELINE_ERROR=13

# =============================================================================
# Logging Functions
# =============================================================================

log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] $SCRIPT_NAME: $1"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] $SCRIPT_NAME: $1" >&2
}

log_fatal() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [FATAL] $SCRIPT_NAME: $1" >&2
}

# =============================================================================
# Error Handling
# =============================================================================

cleanup() {
    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        log_error "Script failed with exit code $exit_code"
        log_error "Check conda environment, project paths, and pipeline logs"
    fi
    exit $exit_code
}

trap cleanup EXIT

# =============================================================================
# Conda Environment Detection and Activation
# =============================================================================

detect_conda_installation() {
    local conda_paths=(
        "$HOME/miniconda3"
        "$HOME/anaconda3"
        "$HOME/miniforge3"
        "/opt/miniconda3"
        "/opt/anaconda3"
        "/usr/local/miniconda3"
        "/usr/local/anaconda3"
    )

    for conda_path in "${conda_paths[@]}"; do
        if [[ -f "$conda_path/bin/conda" ]]; then
            echo "$conda_path"
            return 0
        fi
    done

    # Check if conda is in PATH
    if command -v conda >/dev/null 2>&1; then
        # Try to find conda base from the conda command
        local conda_info
        conda_info=$(conda info --base 2>/dev/null) || return 1
        if [[ -n "$conda_info" && -d "$conda_info" ]]; then
            echo "$conda_info"
            return 0
        fi
    fi

    return 1
}

activate_conda_environment() {
    log_info "Detecting conda installation..."

    local conda_base
    if ! conda_base=$(detect_conda_installation); then
        log_fatal "Could not find conda installation. Checked common paths and PATH."
        exit $EXIT_CONDA_ERROR
    fi

    log_info "Found conda installation at: $conda_base"

    # Source conda setup
    local conda_sh="$conda_base/etc/profile.d/conda.sh"
    if [[ ! -f "$conda_sh" ]]; then
        log_fatal "Conda setup script not found at: $conda_sh"
        exit $EXIT_CONDA_ERROR
    fi

    log_info "Sourcing conda setup from: $conda_sh"
    # shellcheck source=/dev/null
    source "$conda_sh"

    # Check if environment exists
    if ! conda env list | grep -q "^$CONDA_ENV_NAME "; then
        log_fatal "Conda environment '$CONDA_ENV_NAME' not found. Available environments:"
        conda env list >&2
        exit $EXIT_ACTIVATION_ERROR
    fi

    log_info "Activating conda environment: $CONDA_ENV_NAME"
    if ! conda activate "$CONDA_ENV_NAME"; then
        log_fatal "Failed to activate conda environment: $CONDA_ENV_NAME"
        exit $EXIT_ACTIVATION_ERROR
    fi

    log_info "Successfully activated conda environment: $CONDA_ENV_NAME"
    log_info "Python executable: $(which python)"
    log_info "Python version: $(python --version)"
}

# =============================================================================
# Project Path Resolution
# =============================================================================

setup_project_environment() {
    # Detect project root by finding this script's location
    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

    # Navigate up to project root (cron_scripts -> machine_learning -> src -> project_root)
    local project_root
    project_root="$(cd "$script_dir/../../../" && pwd)"

    # Validate project structure - check for test deployment script
    local test_deploy_script="$project_root/src/machine_learning/test_deploy_pipeline.py"
    if [[ ! -f "$test_deploy_script" ]]; then
        log_fatal "Test deploy pipeline script not found at: $test_deploy_script"
        log_fatal "Project root detected as: $project_root"
        exit $EXIT_PROJECT_ERROR
    fi

    log_info "Project root: $project_root"
    log_info "Test deploy script: $test_deploy_script"

    # Change to project root for execution
    cd "$project_root"
    log_info "Changed working directory to: $(pwd)"

    # Export project paths for test_deploy_pipeline.py
    export PROJECT_ROOT="$project_root"
    export PYTHONPATH="$project_root/src:$project_root/src/machine_learning:${PYTHONPATH:-}"
}

# =============================================================================
# Pipeline Execution
# =============================================================================

execute_test_monthly_pipeline() {
    local test_deploy_script="src/machine_learning/test_deploy_pipeline.py"

    log_info "Starting TEST $PIPELINE_MODE pipeline execution (limited to 5 MLBs)..."
    log_info "Command: python $test_deploy_script --mode=$PIPELINE_MODE"

    # Execute the test pipeline with proper error handling
    if python "$test_deploy_script" --mode="$PIPELINE_MODE"; then
        log_info "TEST $PIPELINE_MODE pipeline completed successfully"
        return $EXIT_SUCCESS
    else
        local exit_code=$?
        log_fatal "TEST $PIPELINE_MODE pipeline failed with exit code: $exit_code"
        log_fatal "Check test_deploy_pipeline.py logs for detailed error information"
        return $exit_code
    fi
}

# =============================================================================
# Main Execution
# =============================================================================

main() {
    log_info "============================================================"
    log_info "Starting $SCRIPT_NAME for TEST $PIPELINE_MODE pipeline"
    log_info "NOTE: This is a TEST version limited to 5 MLBs for safe testing"
    log_info "Timestamp: $(date)"
    log_info "User: $(whoami)"
    log_info "Shell: $SHELL"
    log_info "============================================================"

    # Step 1: Activate conda environment
    activate_conda_environment

    # Step 2: Setup project environment and paths
    setup_project_environment

    # Step 3: Execute the test pipeline
    if execute_test_monthly_pipeline; then
        log_info "============================================================"
        log_info "$SCRIPT_NAME completed successfully"
        log_info "============================================================"
        exit $EXIT_SUCCESS
    else
        exit_code=$?
        log_fatal "============================================================"
        log_fatal "$SCRIPT_NAME failed"
        log_fatal "============================================================"
        exit $exit_code
    fi
}

# =============================================================================
# Script Entry Point
# =============================================================================

# Only run main if script is executed directly (not sourced)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
