"""Module 2: follow a model command from Gym to IDEGym, then grade the files.

Run: ./run-experiment 2 run.
Read prepare_task() -> run_task() -> grade_task(), then main() for configuration and reporting.
"""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
import os
import shlex
from collections.abc import Awaitable, Callable
from typing import Optional

import click
from idegym.api.tools.bash import BashCommandResponse
from idegym.client.server import IdeGYMServer
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming

from experiments.artifacts import new_output, record_run
from experiments.bash_access.agent import BashRunRequest, BashVerifyResponse
from experiments.bash_access.cli import BashEvaluationResult
from experiments.bash_access.transport import BashConnection, BashMode, BashSettings, connect_bash
from experiments.common import configure, model_for_provider, prepare_runtime
from experiments.ide_mcp.agent import run_episode
from experiments.ide_mcp.task import GOLD_SCRIPT, PROJECT_PATH, VERIFY_SCRIPT, Grade, parse_grade, python_command
from experiments.ide_mcp.transport import ToolInfo
from experiments.images import BOOKWORM_IMAGE
from experiments.task_files import export_script, parse_submission, restore_script
from experiments.workshop_support import (
    announce,
    concurrent_sandboxes,
    event,
    run_concurrently,
    run_gym,
    run_reference,
    show_checks,
)

MODULE_ROOT = Path(__file__).resolve().parent
TASK_FILES = ("calculator.py", "main.py", "test_calculator.py")


# The direct interface: a model command becomes an actual IDEGym SDK call.
async def execute_direct(server: IdeGYMServer, script: str, timeout: int) -> BashCommandResponse:
    # Preparation supplies Python/pytest; every command starts in the task directory.
    return await server.execute_bash(
        script=f"cd {shlex.quote(PROJECT_PATH)} && "
        "PATH=/opt/idegym/.venv/bin:/usr/local/bin:/usr/bin:/bin "
        f"/bin/bash -c {shlex.quote(script)}",
        command_timeout=timeout,
        max_output_bytes=24000,
    )


# 1. Prepare the task in a fresh generic image before exposing tools to the model.
async def prepare_task(connection: BashConnection, output: Optional[Path] = None) -> list[ToolInfo]:
    tools = await connection.discover()
    if output is not None:
        (output / "tools.json").write_text(json.dumps([tool.model_dump() for tool in tools], indent=2) + "\n")
    await connection.prepare(tools)

    return tools


# 2. Run either the reference repair or the bounded NeMo Gym model/tool loop.
async def run_task(
    connection: BashConnection,
    tools: list[ToolInfo],
    output: Path,
    *,
    params: Optional[NeMoGymResponseCreateParamsNonStreaming] = None,
    model_call: Optional[Callable[[NeMoGymResponseCreateParamsNonStreaming], Awaitable[NeMoGymResponse]]] = None,
    step_limit: int = 20,
) -> tuple[Optional[NeMoGymResponse], int, int]:
    if model_call is None:
        patched = await connection.execute_bash(python_command(GOLD_SCRIPT))
        if patched.exit_code:
            raise RuntimeError("Reference transformation failed")
        return None, 0, 0
    if params is None:
        raise ValueError("Model evaluation requires response parameters")
    return await run_episode(params, tools, connection, model_call, step_limit, output / "trajectory.jsonl")


async def export_submission(connection: BashConnection, output: Path) -> dict[str, str | None]:
    sources = await connection.execute_bash(python_command(export_script(PROJECT_PATH, TASK_FILES)))
    if sources.exit_code:
        raise RuntimeError("Could not export submission")
    submission = parse_submission(sources.stdout, TASK_FILES)
    (output / "final-files.json").write_text(json.dumps(submission, indent=2) + "\n")
    return submission


# 3. Grade only submitted source files in the independently prepared fresh sandbox.
async def grade_task(connection: BashConnection, submission: dict[str, str | None]) -> Grade:
    transferred = await connection.execute_bash(python_command(restore_script(PROJECT_PATH, submission)))
    if transferred.exit_code:
        raise RuntimeError("Could not transfer submission to grading sandbox")
    result = await connection.execute_bash(python_command(VERIFY_SCRIPT))
    return parse_grade(result.stdout, result.exit_code)


async def evaluate(
    settings: BashSettings,
    output: Path,
    *,
    body: Optional[BashRunRequest] = None,
    model_call: Optional[Callable[[NeMoGymResponseCreateParamsNonStreaming], Awaitable[NeMoGymResponse]]] = None,
    step_limit: int = 20,
) -> BashVerifyResponse | BashEvaluationResult:
    """Prepare -> run -> grade; agent and grader start concurrently."""
    golden = model_call is None
    if not golden and body is None:
        raise ValueError("Model evaluation requires a run request")
    event("Starting agent and grading sandboxes concurrently...")
    async with concurrent_sandboxes(connect_bash(settings, output), connect_bash(settings, output)) as sandboxes:
        # 1. Stage a clean task in both generic images concurrently.
        tools, _ = await run_concurrently(prepare_task(sandboxes.agent, output), prepare_task(sandboxes.grader))
        (output / "sandboxes.json").write_text(
            json.dumps(
                {
                    "agent": {"server_id": sandboxes.agent.server.server_id, "image": str(settings.image)},
                    "grader": {"server_id": sandboxes.grader.server.server_id, "image": str(settings.image)},
                },
                indent=2,
            )
            + "\n"
        )
        # 2. Only the agent receives model tools. The grader is ready and waiting.
        event("Sandboxes ready. Starting the bounded bash tool loop...")
        response, attempts, successes = await run_task(
            sandboxes.agent,
            tools,
            output,
            params=None if golden else body.responses_create_params,
            model_call=model_call,
            step_limit=step_limit,
        )
        submission = await export_submission(sandboxes.agent, output)
        await sandboxes.close_agent()
        # 3. The agent is closed; its runtime changes cannot affect grading.
        event("Agent sandbox closed. Grading submitted sources in the fresh sandbox...")
        checks = await grade_task(sandboxes.grader, submission)
    report = BashEvaluationResult(
        reward=checks.reward if golden or successes else 0,
        checks=checks,
        mode=settings.mode,
        golden=golden,
        tool_calls=attempts,
        successful_tool_calls=successes,
    )
    if golden:
        if not checks.reward:
            raise RuntimeError("Reference transformation failed verification")
        return report
    return BashVerifyResponse(**(body.model_dump() | report.model_dump() | {"response": response}))


async def evaluate_reference(settings: BashSettings, output: Path) -> BashEvaluationResult:
    return await evaluate(settings, output)


async def evaluate_bash(
    body: BashRunRequest,
    settings: BashSettings,
    model_call: Callable[[NeMoGymResponseCreateParamsNonStreaming], Awaitable[NeMoGymResponse]],
    step_limit: int,
    output: Path,
) -> BashVerifyResponse:
    # Gym adapter compatibility: the teaching flow lives in evaluate().
    return await evaluate(settings, output, body=body, model_call=model_call, step_limit=step_limit)


# 4. Report scored rollouts; no model parameters are updated in any module.
def report_results(report: BashEvaluationResult, settings: dict, output: Path) -> None:
    if report.mode != settings["mode"] or report.golden != settings["golden"]:
        raise click.ClickException("Rollout mode does not match the requested evaluation")
    (output / "summary.json").write_text(
        json.dumps(report.model_dump() | settings | {"parameter_updates": 0}, indent=2) + "\n"
    )
    show_checks(report.checks, report.reward, output)
    click.echo(f"Successful bash tool calls: {report.successful_tool_calls}/{report.tool_calls}")


def run_workshop(
    module_root: Path,
    mode: BashMode,
    golden: bool,
    model: str | None,
    reasoning_effort: Optional[str],
    step_limit: int,
    output_dir: Optional[Path],
    *,
    run_timeout: int = 1800,
) -> None:
    """Save model traces, final files, and the same five independent reward checks."""
    model = model_for_provider(model or os.environ["POC_MODEL_NAME"])
    image = BOOKWORM_IMAGE
    output = new_output(module_root, output_dir, "golden" if golden else "model")
    settings = dict(
        model=None if golden else model,
        reasoning_effort=reasoning_effort,
        step_limit=step_limit,
        max_output_tokens=4096,
        temperature=0.6,
        mode=mode,
        image=image,
        golden=golden,
        task="calculator-rename",
        transport="OpenEnv EnvClient WebSocket" if mode == "openenv" else "IdeGYM execute_bash HTTP",
    )
    (output / "run-settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    os.environ.update(
        POC_MODEL_NAME=model,
        POC_RUN_DIR=str(output),
        POC_LOG_DIR=str(output / "logs"),
    )
    announce(
        2 if mode == "bash" else 3,
        "Rename calculator operations",
        "calculator-rename",
        settings["transport"],
        model,
        golden,
        step_limit,
        output,
    )
    click.echo("Sandboxes: independent agent and fresh grader; concurrent startup")
    if golden:
        report = run_reference(evaluate_reference(BashSettings(mode=mode, image=image), output), run_timeout, output)
        (output / "rollouts.jsonl").write_text(report.model_dump_json() + "\n")
    else:
        prepare_runtime(("responses_api_agents/bash_agent", "responses_api_models/inference_provider"))
        command = [
            str(Path(sys.executable).with_name("gym")),
            "eval",
            "run",
            "--config",
            str(output / "config.yaml"),
            "--split",
            "validation",
            "--output",
            str(output / "rollouts.jsonl"),
            "--limit",
            "1",
            "--num-repeats",
            "1",
            "--concurrency",
            "1",
            "--max-output-tokens",
            "4096",
            "--temperature",
            "0.6",
            f"++bash_agent.responses_api_agents.bash_agent.step_limit={step_limit}",
        ]
        if reasoning_effort is not None:
            command.append(
                "++policy_model.responses_api_models.inference_provider.extra_body.reasoning_effort="
                + reasoning_effort
            )
        with (output / "gym.log").open("w") as log:
            code = run_gym(command, log, output, timeout_s=run_timeout)
        if code:
            raise click.ClickException(f"Gym exited {code}; inspect {output / 'gym.log'}")
        rows = [json.loads(line) for line in (output / "rollouts.jsonl").read_text().splitlines() if line.strip()]
        if len(rows) != 1:
            raise click.ClickException(f"Expected one rollout, got {len(rows)}")
        report = BashEvaluationResult.model_validate(rows[0])
    report_results(report, settings, output)


@click.command()
@click.option("--golden", is_flag=True, help="Validate this transport and verifier without inference.")
@click.option("--model", envvar="POC_MODEL_NAME", help="Override the selected provider model.")
@click.option("--reasoning-effort", type=click.Choice(["none", "low", "medium", "high", "xhigh", "max"]))
@click.option("--step-limit", default=20, type=click.IntRange(1, 100), show_default=True)
@click.option(
    "--run-timeout",
    default=1800,
    type=click.IntRange(1),
    show_default=True,
    help="Maximum seconds for the complete evaluation, including sandbox startup and grading.",
)
@click.option("--output-dir", type=click.Path(path_type=Path))
@record_run(MODULE_ROOT)
def main(
    golden: bool,
    model: str,
    reasoning_effort: Optional[str],
    step_limit: int,
    run_timeout: int,
    output_dir: Optional[Path],
) -> None:
    """Rename calculator operations through IDEGym's bash interface."""
    configure(MODULE_ROOT)
    run_workshop(MODULE_ROOT, "bash", golden, model, reasoning_effort, step_limit, output_dir, run_timeout=run_timeout)


if __name__ == "__main__":
    main()
