"""All the general configuration of the project."""

from pathlib import Path

SRC = Path(__file__).resolve()

ROOT = SRC.joinpath("..", "..").resolve()

BLD = ROOT.joinpath("bld").resolve()

DATA = ROOT.joinpath("data").resolve()
