"""Module 4: model-visible IDE diagnostics, then independent grading in a new sandbox.

Run: ./run-experiment 4 run.
Read prepare_task() -> run_task() -> grade_task(), then main() for configuration and reporting.
"""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import asyncio
import json
import os
from collections.abc import Awaitable, Callable
from time import monotonic

import click
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming

from experiments.artifacts import new_output, record_run
from experiments.common import configure, model_for_provider, prepare_runtime
from experiments.ide_mcp.agent import run_episode
from experiments.ide_mcp.task import PROJECT_PATH, python_command
from experiments.ide_mcp.transport import IdeConnection, IdeSettings, ToolInfo, connect_ide
from experiments.images import PYCHARM_IMAGE
from experiments.module_4_jetbrains_mcp.evaluation import (
    MODEL_TOOLS,
    BehaviorChecks,
    InspectionChecks,
    InspectionResult,
    ScopedInspectionConnection,
    inspected_before_and_after_edits,
)
from experiments.module_4_jetbrains_mcp.reporting import write_summary
from experiments.module_4_jetbrains_mcp.task import (
    BROKEN,
    FILES,
    FIXED,
    VERIFY_SCRIPT,
    inspect,
    seed,
    write_files_script,
)
from experiments.task_files import export_script, parse_submission
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


# 1. Prepare runtime files and wait until PyCharm sees the expected diagnostics.
async def prepare_task(
    connection: IdeConnection, settings: IdeSettings, output: Path, *, phase: str = "before"
) -> tuple[list[ToolInfo], dict[str, list[dict]]]:
    tools = await connection.discover()
    if "pycharm_get_file_problems" not in {tool.name for tool in tools}:
        raise RuntimeError("Image does not expose the integrated inspection tool")
    if phase == "before":
        (output / "tools.json").write_text(json.dumps([t.model_dump() for t in tools], indent=2) + "\n")
        (output / "model-tools.json").write_text(
            json.dumps([t.model_dump() for t in tools if t.name in MODEL_TOOLS], indent=2) + "\n"
        )
    await seed(connection)
    if phase == "before":
        (output / "initial-files.json").write_text(json.dumps(BROKEN, indent=2) + "\n")
    deadline = monotonic() + settings.ready_timeout
    while True:
        before = await inspect(connection, output, phase)
        if all(before.values()):
            break
        if monotonic() >= deadline:
            raise RuntimeError(f"{phase} baseline invalid: PyCharm did not detect seeded problems in both files")
        await asyncio.sleep(3)
    return tools, before


# 2. Run the reference repair or a model loop exposing only the scoped MCP tools.
async def run_task(
    connection: IdeConnection,
    settings: IdeSettings,
    tools: list[ToolInfo],
    output: Path,
    *,
    params: NeMoGymResponseCreateParamsNonStreaming | None = None,
    model_call: Callable[[NeMoGymResponseCreateParamsNonStreaming], Awaitable[NeMoGymResponse]] | None = None,
    step_limit: int = 20,
) -> tuple[NeMoGymResponse | None, int, int]:
    if model_call is None:
        patched = await connection.server.execute_bash(
            script=python_command(write_files_script(FIXED)),
            command_timeout=30,
        )
        if patched.exit_code:
            raise RuntimeError("Reference repair failed")
        return None, 0, 0
    else:
        if params is None:
            raise ValueError("Model evaluation requires response parameters")
        # 2. Inspections are model observations, not just a post-hoc score.
        # Successful inspection calls are required before AND after edits.
        event("Starting the inspection -> edit -> inspection tool loop...")
        response, attempts, successes = await run_episode(
            params,
            [tool for tool in tools if tool.name in MODEL_TOOLS],
            ScopedInspectionConnection(connection.session, connection.server, connection.scope["client_id"], settings),
            model_call,
            step_limit,
            output / "trajectory.jsonl",
        )
    return response, attempts, successes


async def export_submission(connection: IdeConnection, output: Path) -> dict[str, str | None]:
    # Transfer only the two submitted sources, with bounds checked before grading.
    sources = await connection.server.execute_bash(
        script=python_command(export_script(PROJECT_PATH, FILES)), command_timeout=30, max_output_bytes=1048576
    )
    if sources.exit_code:
        raise RuntimeError("Could not export submission")
    submission = parse_submission(sources.stdout, FILES)
    (output / "final-files.json").write_text(json.dumps(submission, indent=2) + "\n")
    return submission


# 3. Grade after closing the agent sandbox; only source files enter this fresh sandbox.
async def grade_task(
    grader: IdeConnection, output: Path, submission: dict[str, str | None]
) -> tuple[dict[str, list[dict]], dict[str, bool]]:
    event("Agent sandbox closed. Grading in the prepared fresh PyCharm sandbox...")
    transferred = await grader.server.execute_bash(
        script=python_command(
            write_files_script({name: source for name, source in submission.items() if source is not None}, clean=True)
        ),
        command_timeout=30,
    )
    if transferred.exit_code:
        raise RuntimeError("Could not transfer submission to grading sandbox")
    present = tuple(name for name in FILES if submission[name] is not None)
    after = await inspect(grader, output, "after", files=present)
    for name in FILES:
        if submission[name] is None:
            after[name] = [{"severity": "ERROR", "description": "Submitted file is missing"}]
    (output / "inspections-after.json").write_text(json.dumps(after, indent=2) + "\n")
    behavior_result = await grader.server.execute_bash(
        script=python_command(VERIFY_SCRIPT),
        command_timeout=30,
    )
    if behavior_result.exit_code:
        raise RuntimeError("Behavior verifier did not execute")
    behavior = BehaviorChecks.model_validate_json(behavior_result.stdout, strict=True).model_dump()
    return after, behavior


def save_verification(result: InspectionResult, output: Path) -> None:
    (output / "verification.json").write_text(result.model_dump_json(indent=2) + "\n")


async def evaluate(
    settings: IdeSettings,
    output: Path,
    *,
    params: NeMoGymResponseCreateParamsNonStreaming | None = None,
    model_call: Callable[[NeMoGymResponseCreateParamsNonStreaming], Awaitable[NeMoGymResponse]] | None = None,
    step_limit: int = 20,
) -> tuple[InspectionResult, NeMoGymResponse | None]:
    """Prepare -> run -> grade; agent and grader start concurrently."""
    golden = model_call is None
    event("Starting agent and grading PyCharm sandboxes concurrently...")
    async with concurrent_sandboxes(connect_ide(settings), connect_ide(settings)) as sandboxes:
        # 1. Stage runtime files and validate both IDE baselines concurrently.
        (tools, before), (_, grader_before) = await run_concurrently(
            prepare_task(sandboxes.agent, settings, output),
            prepare_task(sandboxes.grader, settings, output, phase="grader-before"),
        )
        identities = {
            "agent": sandboxes.agent.scope | {"image": str(settings.image)},
            "grader": sandboxes.grader.scope | {"image": str(settings.image)},
        }
        (output / "sandboxes.json").write_text(json.dumps(identities, indent=2) + "\n")
        # 2. Only the agent exposes MCP tools to the model; the grader waits.
        response, attempts, successes = await run_task(
            sandboxes.agent, settings, tools, output, params=params, model_call=model_call, step_limit=step_limit
        )
        submission = await export_submission(sandboxes.agent, output)
        await sandboxes.close_agent()
        # 3. Transfer only submitted sources into the independently prepared grader.
        after, behavior = await grade_task(sandboxes.grader, output, submission)
    event("Fresh sandbox grading finished; both sandboxes closed.")
    used = golden
    if not golden:
        entries = [json.loads(line) for line in (output / "trajectory.jsonl").read_text().splitlines()]
        used = inspected_before_and_after_edits(entries)
    checks = InspectionChecks(
        baseline_detected=all(bool(problems) for problems in before.values())
        and all(bool(problems) for problems in grader_before.values()),
        inspections_clean=not any(after.values()),
        behavior_preserved=all(behavior.values()),
        model_used_inspections=used,
    )
    result = InspectionResult(
        reward=checks.reward,
        checks=checks,
        golden=golden,
        before_counts={name: len(problems) for name, problems in before.items()},
        after_counts={name: len(problems) for name, problems in after.items()},
        behavior=behavior,
        tool_calls=attempts,
        successful_tool_calls=successes,
    )
    save_verification(result, output)
    return result, response


# 4. Report scored rollouts; no model parameters are updated in any module.
def report_results(output: Path, settings: dict) -> None:
    result = write_summary(output, settings)
    show_checks(result.checks, result.reward, output)
    for name in result.before_counts:
        click.echo(f"  {name}: {result.before_counts[name]} diagnostics before -> {result.after_counts[name]} after")
    click.echo(f"Report: {output / 'README.md'}")


# CLI: configure -> run through NeMo Gym (or reference) -> report.
@click.command()
@click.option("--golden", is_flag=True, help="Validate the reference fix without model calls.")
@click.option("--model", envvar="POC_MODEL_NAME", help="Override the selected provider model.")
@click.option(
    "--reasoning-effort",
    default="omit",
    type=click.Choice(["omit", "none", "low", "medium", "high"]),
    show_default=True,
)
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
    reasoning_effort: str,
    step_limit: int,
    run_timeout: int,
    output_dir: Path | None,
) -> None:
    """Save model traces, independent inspection snapshots, and reward results."""
    configure(MODULE_ROOT)
    model = model_for_provider(model or os.environ["POC_MODEL_NAME"])
    reasoning_effort = None if reasoning_effort == "omit" else reasoning_effort
    image = PYCHARM_IMAGE
    output = new_output(MODULE_ROOT, output_dir, "golden" if golden else "model")
    settings = dict(
        image=image,
        model=None if golden else model,
        reasoning_effort=reasoning_effort,
        step_limit=step_limit,
        temperature=0.6,
        golden=golden,
        fixture="pricing-receipt-v1",
        parameter_updates=0,
    )
    (output / "run-settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    announce(
        4,
        "Repair IDE inspections",
        "pricing-receipt-v1",
        "PyCharm inspection MCP tools; separate agent and grading sandboxes",
        model,
        golden,
        step_limit,
        output,
    )
    click.echo("Sandboxes: independent agent and fresh grader; concurrent startup")
    os.environ.update(POC_MODEL_NAME=model, POC_RUN_DIR=str(output), POC_LOG_DIR=str(output / "logs"))
    if golden:
        result, _ = run_reference(evaluate(IdeSettings(mode="bundled", image=image), output), run_timeout, output)
        (output / "rollouts.jsonl").write_text(result.model_dump_json() + "\n")
    else:
        prepare_runtime(("responses_api_agents/inspection_agent", "responses_api_models/inference_provider"))
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
            f"++ide_agent.responses_api_agents.inspection_agent.step_limit={step_limit}",
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
    report_results(output, settings)


if __name__ == "__main__":
    main()
