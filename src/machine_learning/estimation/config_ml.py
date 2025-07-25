"""All the general configuration of the project."""

from pathlib import Path

# Get the path to this file and resolve it
SRC = Path(__file__).parent.resolve()

# Navigate up to the machine_learning directory, then to root
ROOT = SRC.parent.parent.resolve()

# Build and data directories
BLD = ROOT / "bld"
DATA = ROOT / "data"

# Ensure directories exist
BLD.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)
