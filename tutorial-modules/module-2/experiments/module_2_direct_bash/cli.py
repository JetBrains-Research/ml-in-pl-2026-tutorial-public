"""Module 2: bash access."""

from pathlib import Path

from experiments.bash_access.cli import create_cli

cli = create_cli(Path(__file__).resolve().parent)
