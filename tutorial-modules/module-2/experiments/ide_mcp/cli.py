"""Click commands for the bundled PyCharm MCP module."""

import asyncio
import json
from pathlib import Path
from time import monotonic

import click

from experiments.artifacts import new_output
from experiments.common import configure
from experiments.ide_mcp.task import Mode
from experiments.ide_mcp.transport import IdeSettings, connect_ide
from experiments.image_build import build_image as build_cloud_image
from experiments.images import PYCHARM_IMAGE


async def check_image(settings: IdeSettings, output: Path) -> None:
    from experiments.module_4_jetbrains_mcp.task import inspect, seed

    async with connect_ide(settings) as connection:
        tools = await connection.discover()
        (output / "tools.json").write_text(json.dumps([tool.model_dump() for tool in tools], indent=2) + "\n")
        if "pycharm_get_file_problems" not in {tool.name for tool in tools}:
            raise RuntimeError("Image does not expose the integrated inspection tool")
        await seed(connection)
        deadline = monotonic() + settings.ready_timeout
        while True:
            problems = await inspect(connection, output, "before")
            if all(problems.values()):
                break
            if monotonic() >= deadline:
                raise RuntimeError("Baseline invalid: PyCharm did not detect seeded problems in both files")
            await asyncio.sleep(3)
    click.echo(f"IDE ready: {len(tools)} {settings.mode} tools; sandbox deleted. Artifacts: {output}")


def create_cli(module_root: Path, mode: Mode) -> click.Group:
    @click.group()
    @click.pass_context
    def cli(ctx: click.Context) -> None:
        """Build a PyCharm image, check inspections, or run the inspection workshop."""
        if ctx.invoked_subcommand != "run":
            configure(module_root)

    @cli.command("build-image")
    @click.option("--job", help="Resume polling an existing IDEGym cloud build job.")
    def build_image(job: str | None) -> None:
        """Build and push the generic PyCharm runtime through IDEGym Cloud Build."""
        asyncio.run(build_cloud_image(module_root=module_root, job_name=job, metadata={"runtime": "pycharm-mcp"}))

    @cli.command()
    def check() -> None:
        """Start a sandbox, discover the selected MCP tools, and check the task fixture."""
        settings = IdeSettings(mode=mode, image=PYCHARM_IMAGE)
        output = new_output(module_root, None, "check")
        asyncio.run(check_image(settings, output))

    from experiments.module_4_jetbrains_mcp.workshop import main

    cli.add_command(main, "run")

    return cli
