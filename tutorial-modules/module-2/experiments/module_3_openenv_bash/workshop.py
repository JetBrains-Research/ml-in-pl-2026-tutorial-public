"""Module 3: official Tbench2Env reset -> Tbench2Action -> step -> observation.

Run: ./run-experiment 3 run. The server image owns the environment; this
workshop supplies task setup, a NeMo Gym episode, and verification commands.
Read prepare_task() -> run_task() -> grade_task(), then main() for configuration and reporting.
"""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import socket
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import click

from experiments.artifacts import new_output, record_run
from experiments.images import OPENENV_IMAGE
from experiments.module_3_openenv_bash.task import (
    PROJECT_PATH,
    checks_from_output,
    reference_command,
    runtime_setup,
    verifier_setup,
)
from experiments.task_files import export_script, parse_submission, restore_script

if TYPE_CHECKING:
    from collections.abc import Awaitable

    from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
    from openenv.core.client_types import StepResult
    from tbench2_env import Tbench2Observation

    from experiments.ide_mcp.task import Grade
    from experiments.ide_mcp.transport import ToolInfo
    from experiments.module_3_openenv_bash.agent import (
        TerminalEvaluationResult,
        TerminalResult,
        TerminalRunRequest,
    )
    from experiments.module_3_openenv_bash.client import WorkshopTerminalEnv
    from experiments.module_3_openenv_bash.transport import OpenEnvBridge, TerminalSettings

MODULE_ROOT = Path(__file__).resolve().parent
TASK_FILES = ("calculator.py", "main.py", "test_calculator.py")


# 1. Prepare runtime files through the official OpenEnv interface.
async def prepare_task(
    client: WorkshopTerminalEnv,
    reset: dict[str, str],
    *,
    setup_command: Optional[str] = None,
    bootstrap_path: str = "/tmp",
) -> StepResult[Tbench2Observation]:
    from tbench2_env import Tbench2Action

    # Local TB2 accepts an existing directory with no instruction or verifier.
    # Bootstrap in /tmp so a generic server can execute our setup command.
    # These calls belong to preparation, before the model sees any tools.
    bootstrap = await client.reset(task_id="workshop-bootstrap", task_path=bootstrap_path)
    if not bootstrap.observation.success or bootstrap.done:
        raise RuntimeError(f"Official tbench2_env bootstrap failed: {bootstrap.observation.error}")
    staged = await client.step(Tbench2Action(action_type="exec", command=setup_command or runtime_setup()))
    if not staged.observation.success:
        raise RuntimeError(f"Runtime task setup failed: {staged.observation.error}")
    # Select the staged task without grading files. The client retains the
    # verifier until the model finishes; no task is baked in the image.
    initial = await client.reset(**reset)
    if not initial.observation.success or initial.done:
        raise RuntimeError(f"Official tbench2_env reset failed: {initial.observation.error}")
    hidden = await client.step(
        Tbench2Action(action_type="exec", command=f"test ! -e {shlex.quote(reset['task_path'] + '/tests')}")
    )
    if not hidden.observation.success:
        raise RuntimeError("Runtime setup exposed verifier files before the model episode")

    return initial


# This function runs in the OpenEnv worker process.
async def run_client(
    base_url: str,
    reset: dict[str, str],
    read_line: Callable[[], str],
    emit: Callable[[dict[str, Any]], None],
    setup_command: Optional[str] = None,
    bootstrap_path: str = "/tmp",
) -> None:
    # Import the official environment client and its action type. Authentication
    # is the only transport adaptation; the environment implementation is upstream.
    from tbench2_env import Tbench2Action

    from experiments.module_3_openenv_bash.client import WorkshopTerminalEnv

    async with WorkshopTerminalEnv(base_url=base_url, message_timeout_s=150) as client:
        initial = await prepare_task(client, reset, setup_command=setup_command, bootstrap_path=bootstrap_path)
        # TB2 has a typed action interface, not MCP tool discovery. The harness
        # exposes one bash function, whose command field comes from that schema.
        command_schema = Tbench2Action.model_json_schema()["properties"]["command"]
        tools = [
            {
                "name": "bash",
                "description": "Run a shell command inside the calculator's IDEGym container.",
                "input_schema": {
                    "type": "object",
                    "properties": {"command": command_schema},
                    "required": ["command"],
                    "additionalProperties": False,
                },
            }
        ]
        emit({"reset": initial.observation.model_dump(), "tools": tools})
        while line := await asyncio.to_thread(read_line):
            request = json.loads(line)
            if request["op"] == "state":
                emit((await client.state()).model_dump())
                continue
            # A model's bash call becomes exec; only the harness sends evaluate.
            # The official local backend performs both actions in this container.
            action = Tbench2Action.model_validate(request["action"])
            result = await client.step(action)
            emit({"observation": result.observation.model_dump(), "reward": result.reward, "done": result.done})


# 2. Run the reference repair or the same bounded model/tool loop as Modules 2 and 4.
async def run_task(
    connection: OpenEnvBridge,
    tools: list[ToolInfo],
    output: Path,
    *,
    params: Optional[NeMoGymResponseCreateParamsNonStreaming] = None,
    model_call: Optional[Callable[[NeMoGymResponseCreateParamsNonStreaming], Awaitable[NeMoGymResponse]]] = None,
    step_limit: int = 20,
) -> tuple[Optional[NeMoGymResponse], int, int]:
    from experiments.ide_mcp.agent import run_episode

    if model_call is None:
        patched = await connection.call("bash", {"command": reference_command()})
        if patched.is_error:
            raise RuntimeError("Reference transformation failed in tbench2_env")
        return None, 0, 0
    if params is None:
        raise ValueError("Model evaluation requires response parameters")
    return await run_episode(params, tools, connection, model_call, step_limit, output / "trajectory.jsonl")


async def export_submission(connection: OpenEnvBridge, output: Path) -> dict[str, Optional[str]]:
    from experiments.module_3_openenv_bash.task import python_command

    files = await connection.call("bash", {"command": python_command(export_script(PROJECT_PATH, TASK_FILES))})
    if files.is_error:
        raise RuntimeError("Could not export the submitted project files")
    text = "\n".join(block["text"] for block in files.content if block.get("type") == "text")
    submission = parse_submission(text, TASK_FILES)
    (output / "final-files.json").write_text(json.dumps(submission, indent=2) + "\n")
    return submission


# 3. Supply only submitted sources and the trusted verifier to a fresh OpenEnv sandbox.
async def grade_task(
    connection: OpenEnvBridge, output: Path, submission: dict[str, Optional[str]]
) -> tuple[Grade, float]:
    from experiments.module_3_openenv_bash.task import python_command
    from experiments.workshop_support import event

    transferred = await connection.call("bash", {"command": python_command(restore_script(PROJECT_PATH, submission))})
    if transferred.is_error:
        raise RuntimeError("Could not transfer submission to grading sandbox")
    # TB2 stages the trusted task tests, runs tests/test.sh, and reads the
    # verifier reward. The model only has bash; grading is a harness action.
    event("Evaluating with tbench2_env's trusted task verifier...")
    staged = await connection.call("bash", {"command": verifier_setup()})
    if staged.is_error:
        raise RuntimeError("Could not stage the trusted verifier after the model episode")
    evaluation = await connection.evaluate()
    (output / "evaluation.json").write_text(json.dumps(evaluation, indent=2) + "\n")
    observation = evaluation["observation"]
    if not observation["success"] or not evaluation["done"]:
        raise RuntimeError(f"The official environment could not verify the submission: {observation['error']}")
    checks = checks_from_output(observation["output"])
    environment_reward = evaluation["reward"]
    if environment_reward is None or environment_reward != checks.reward:
        raise RuntimeError("The official verification reward is missing or disagrees with its checks")
    state = await connection.state()
    (output / "environment-state.json").write_text(json.dumps(state, indent=2) + "\n")
    return checks, environment_reward


async def evaluate(
    settings: TerminalSettings,
    output: Path,
    *,
    body: Optional[TerminalRunRequest] = None,
    model_call: Optional[Callable[[NeMoGymResponseCreateParamsNonStreaming], Awaitable[NeMoGymResponse]]] = None,
    step_limit: int = 20,
) -> TerminalResult | TerminalEvaluationResult:
    from experiments.module_3_openenv_bash.agent import TerminalEvaluationResult, TerminalResult
    from experiments.module_3_openenv_bash.transport import connect_openenv
    from experiments.workshop_support import concurrent_sandboxes, event

    golden = model_call is None
    if not golden and body is None:
        raise ValueError("Model evaluation requires a run request")
    grader_output = output / "grader"
    grader_output.mkdir()
    event("Starting agent and fresh grading OpenEnv sandboxes concurrently...")
    async with concurrent_sandboxes(
        connect_openenv(settings, output), connect_openenv(settings, grader_output)
    ) as sandboxes:
        # 1. Both isolated workers call prepare_task() before their connections return.
        # OpenEnv and Gym share the locked project Python environment.
        tools = await sandboxes.agent.discover()
        (output / "tools.json").write_text(json.dumps([tool.model_dump() for tool in tools], indent=2) + "\n")
        (output / "sandboxes.json").write_text(
            json.dumps(
                {
                    "agent": sandboxes.agent.scope,
                    "grader": sandboxes.grader.scope,
                },
                indent=2,
            )
            + "\n"
        )
        event("OpenEnv ready: bash maps to Tbench2Action(action_type='exec').")
        # 2. The model has access only to the agent's OpenEnv session.
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
        # 3. The prepared grader receives sources and verifier only after the model ends.
        event("Agent sandbox closed. Evaluating in the fresh OpenEnv sandbox...")
        checks, environment_reward = await grade_task(sandboxes.grader, output, submission)

    report = TerminalEvaluationResult(
        reward=checks.reward if golden or successes else 0,
        checks=checks,
        golden=golden,
        tool_calls=attempts,
        successful_tool_calls=successes,
        environment_reward=environment_reward,
    )
    if golden:
        return report
    return TerminalResult(**(body.model_dump() | report.model_dump() | {"response": response}))


# 4. Report scored rollouts; no model parameters are updated in any module.
def report_results(report: TerminalEvaluationResult, settings: dict, output: Path) -> None:
    from experiments.workshop_support import show_checks

    (output / "summary.json").write_text(
        json.dumps(report.model_dump() | settings | {"parameter_updates": 0}, indent=2) + "\n"
    )
    show_checks(report.checks, report.reward, output)
    click.echo(f"Official environment reward: {report.environment_reward:.3f}")
    click.echo(f"Successful model tool calls: {report.successful_tool_calls}/{report.tool_calls}")


# CLI: configure -> run through NeMo Gym (or reference) -> report.
@click.command()
@click.option("--golden", is_flag=True, help="Check official reset, bash actions, and grading without inference.")
@click.option("--model", envvar="POC_MODEL_NAME", help="Override the selected provider model.")
@click.option("--reasoning-effort", type=click.Choice(["none", "low", "medium", "high", "xhigh", "max"]))
@click.option("--step-limit", default=20, type=click.IntRange(1, 100), show_default=True)
@click.option("--run-timeout", default=1800, type=click.IntRange(1), show_default=True)
@click.option("--output-dir", type=click.Path(path_type=Path))
@record_run(MODULE_ROOT)
def main(
    golden: bool,
    model: Optional[str],
    reasoning_effort: Optional[str],
    step_limit: int,
    run_timeout: int,
    output_dir: Optional[Path],
) -> None:
    """Run the calculator exercise in the official OpenEnv tbench2_env."""
    from experiments.common import configure, model_for_provider, prepare_runtime
    from experiments.module_3_openenv_bash.agent import TerminalEvaluationResult
    from experiments.module_3_openenv_bash.transport import TerminalSettings
    from experiments.workshop_support import announce, run_gym, run_reference

    configure(MODULE_ROOT)
    image = OPENENV_IMAGE
    model = model_for_provider(model or os.environ["POC_MODEL_NAME"])
    output = new_output(MODULE_ROOT, output_dir, "golden" if golden else "model")
    settings = dict(
        model=None if golden else model,
        reasoning_effort=reasoning_effort,
        step_limit=step_limit,
        max_output_tokens=4096,
        temperature=0.6,
        mode="openenv",
        execution_backend="local",
        image=image,
        golden=golden,
        task="calculator-rename",
        transport="Official Tbench2Env / Tbench2Action over OpenEnv WebSocket",
        orchestrator_url=os.environ.get("IDEGYM_ORCHESTRATOR_URL"),
        environment_source="https://github.com/huggingface/OpenEnv/tree/8d3455ab91d28b46349b8bf01fc243d339cda260/envs/tbench2_env",
    )
    (output / "run-settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    os.environ.update(
        POC_MODEL_NAME=model,
        POC_RUN_DIR=str(output),
        POC_LOG_DIR=str(output / "logs"),
    )
    announce(
        3,
        "Refactor with official OpenEnv bash",
        "calculator-rename",
        settings["transport"],
        model,
        golden,
        step_limit,
        output,
    )
    click.echo(f"IDEGym orchestrator: {settings['orchestrator_url']}")
    click.echo(f"Environment image: {image}")
    click.echo("Execution: shell commands inside the IDEGym container (TB2_MODE=local)")
    click.echo("Sandboxes: independent agent and fresh grader; concurrent startup")
    if golden:
        try:
            report = run_reference(evaluate(TerminalSettings(image=image), output), run_timeout, output)
        except RuntimeError as error:
            raise click.ClickException(str(error)) from error
        (output / "rollouts.jsonl").write_text(report.model_dump_json() + "\n")
    else:
        prepare_runtime(("responses_api_agents/bash_agent", "responses_api_models/inference_provider"))
        # An independent head port isolates this run from another local Gym run.
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            head_port = listener.getsockname()[1]
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
            f"++head_server.port={head_port}",
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
        rows = [line for line in (output / "rollouts.jsonl").read_text().splitlines() if line.strip()]
        if len(rows) != 1:
            raise click.ClickException(f"Expected one rollout, got {len(rows)}")
        report = TerminalEvaluationResult.model_validate_json(rows[0])
    report_results(report, settings, output)


if __name__ == "__main__":
    main()
