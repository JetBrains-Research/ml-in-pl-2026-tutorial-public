"""Run the fixed official OpenEnv terminal image."""

import click

from experiments.module_3_openenv_bash.workshop import main


@click.group()
def cli() -> None:
    """Run the official OpenEnv bash exercise."""


cli.add_command(main, "run")
