"""Run the calculator exercise on the shared generic Bookworm image."""

from pathlib import Path

import click
from pydantic import BaseModel, Field, model_validator

from experiments.bash_access.transport import BashMode
from experiments.ide_mcp.task import Grade


class BashEvaluationResult(BaseModel):
    reward: float
    checks: Grade
    mode: BashMode
    golden: bool = False
    tool_calls: int = Field(ge=0)
    successful_tool_calls: int = Field(ge=0)

    @model_validator(mode="after")
    def check_reward(self) -> "BashEvaluationResult":
        if self.reward != (self.checks.reward if self.golden or self.successful_tool_calls else 0):
            raise ValueError("Reward disagrees with the checks or bash interaction count")
        return self


def create_cli(module_root: Path) -> click.Group:
    @click.group()
    def cli() -> None:
        """Run the calculator task with direct bash."""

    # The workshop owns the runnable teaching path.
    from experiments.module_2_direct_bash.workshop import main as direct_main

    cli.add_command(direct_main, "run")

    return cli
