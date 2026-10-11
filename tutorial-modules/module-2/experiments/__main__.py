"""One numbered entry point for the NeMo Gym tutorial experiments."""

from importlib import import_module

import click

MODULES = {
    "1": ("module_1_python", "Tiny Python"),
    "2": ("module_2_direct_bash", "Direct bash"),
    "3": ("module_3_openenv_bash", "Official OpenEnv bash"),
    "4": ("module_4_jetbrains_mcp", "PyCharm MCP inspection repair"),
}


class ExperimentCLI(click.Group):
    def list_commands(self, ctx: click.Context) -> list[str]:
        return list(MODULES)

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        module = MODULES.get(cmd_name)
        if module is None:
            return None
        return import_module(f"experiments.{module[0]}.cli").cli

    def format_commands(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        with formatter.section("Experiments"):
            formatter.write_dl([(number, label) for number, (_, label) in MODULES.items()])


@click.group(cls=ExperimentCLI)
def cli() -> None:
    """Run an experiment: NUMBER run [OPTIONS]."""


if __name__ == "__main__":
    cli()
