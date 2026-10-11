"""Run the PyCharm inspection workshop or prepare its server image."""

from pathlib import Path

from experiments.ide_mcp.cli import create_cli

MODULE_ROOT = Path(__file__).resolve().parent
cli = create_cli(MODULE_ROOT, "bundled")
