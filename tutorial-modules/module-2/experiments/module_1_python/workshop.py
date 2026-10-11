# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Module 1: a model writes Python, NeMo Gym evaluates it, and IDEGym executes it.

Read prepare_task() -> run_task() -> grade_task(), then main() for configuration and reporting.
Gym generates the answer first; this module's run_task() executes that answer without a tool loop.
Run from tutorial-modules/module-2/: ./run-experiment 1 run

Flow: sample prompt -> Gym SimpleAgent -> hosted model -> our verifier -> IDEGym.
main() launches Gym; its resources-server callback delegates to evaluate().
The generated solution executes in the remote sandbox, not in this process.
"""

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlsplit

# Allow learners to run this file directly as well as through the numbered CLI.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import click
from nemo_gym.sandbox import AsyncSandbox, SandboxExecResult, SandboxSpec
from omegaconf import DictConfig, OmegaConf
from pydantic import ValidationError

from experiments.artifacts import new_output, record_run, save_configuration
from experiments.common import configure, model_for_provider, prepare_runtime
from experiments.images import BOOKWORM_IMAGE
from experiments.module_1_python.workshop_support import (
    PythonResourcesConfig,
    PythonSample,
    PythonVerifyRequest,
    PythonVerifyResponse,
    TaskMetadata,
    extract_code,
    output_matches,
    progress,
    run_with_progress,
    show_result,
    test_command,
)

MODULE_ROOT = Path(__file__).resolve().parent

SAMPLES_FILE = MODULE_ROOT / "data/sample.jsonl"
# These limits govern different stages: remote test execution and model generation.
TEST_TIMEOUT_S = 5
MAX_OUTPUT_TOKENS = 512

IMAGE = BOOKWORM_IMAGE


# Task data: each file row holds a prompt and separate verifier-only test answers.
def load_samples(path: Path, limit: int) -> list[PythonSample]:
    # responses_create_params contains the model input. verifier_metadata contains
    # test inputs and expected answers; SimpleAgent does not put it in the prompt.
    samples = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            samples.append(PythonSample.model_validate_json(line))
        except ValidationError as exc:
            raise click.ClickException(
                f"Invalid sample in {path} at line {line_number}: "
                "each row needs responses_create_params and verifier_metadata with test_cases"
            ) from exc
    if not samples:
        raise click.ClickException(f"No samples found in {path}")
    if limit > len(samples):
        raise click.ClickException(f"Requested {limit} samples, but {path} contains only {len(samples)}")
    # Validate the whole file before starting services, then select rows in file order.
    return samples[:limit]


# NeMo Gym wiring: SimpleAgent calls the model, then our verifier.
def gym_configuration(dataset: Path, test_timeout: int) -> DictConfig:
    # Keep connection/runtime settings separate from the service wiring taught here.
    config = OmegaConf.load(MODULE_ROOT / "config.yaml")
    # A resources server supplies environment operations and grading to Gym.
    # Here its app.py adapter delegates grading to evaluate() in this file.
    config.idegym_python = {
        "resources_servers": {
            "idegym_python": {
                "entrypoint": "app.py",
                "domain": "coding",
                "verified": False,
                "description": "Execute generated Python in IDEGym and check its answers",
                "image": IMAGE,
                "concurrency": 1,
                "test_timeout_s": test_timeout,
            }
        }
    }
    # These references connect the agent to the model and verifier services by name.
    # With no model tools and max_steps=1, each response is graded without a repair loop.
    config.idegym_python_agent = {
        "responses_api_agents": {
            "simple_agent": {
                "entrypoint": "app.py",
                "model_server": {"type": "responses_api_models", "name": "policy_model"},
                "resources_server": {"type": "resources_servers", "name": "idegym_python"},
                "max_steps": 1,
                "datasets": [
                    {
                        "name": "workshop_python",
                        "type": "validation",
                        "jsonl_fpath": str(dataset),
                        "license": "Apache 2.0",
                    }
                ],
            }
        }
    }
    # Gym's inference adapter translates its Responses API requests into the hosted
    # provider's chat-completions API. Environment references keep secrets out of snapshots.
    config.policy_model = {
        "responses_api_models": {
            "inference_provider": {
                "entrypoint": "app.py",
                "base_url": "${oc.env:POC_MODEL_BASE_URL}",
                "api_key": "${oc.env:POC_MODEL_API_KEY}",
                "model": "${oc.env:POC_MODEL_NAME}",
            }
        }
    }
    return config


# 1. Prepare a generic sandbox with the generated answer as a runtime file.
def prepare_task(code: str, task: TaskMetadata, config: PythonResourcesConfig) -> SandboxSpec:
    # Expected answers stay on the client; only solution.py is uploaded.
    return SandboxSpec(
        image=config.image,
        workdir="/tmp",
        files={"/tmp/solution.py": code},
        resources={"cpu": 1, "memory_mib": 2048, "disk_gib": 2},
        ready_timeout_s=config.ready_timeout_s,
        metadata={"instance_id": task.task_id},
    )


# 2. Execute the model's one-shot answer for each input through NeMo Gym's sandbox API.
async def run_task(
    sandbox: AsyncSandbox, task: TaskMetadata, config: PythonResourcesConfig
) -> list[SandboxExecResult]:
    # Gym SimpleAgent has already generated the answer; the model has no tools here.
    preflight = await sandbox.exec(shlex.join([config.python_executable, "--version"]), timeout_s=15)
    if preflight.return_code != 0 or preflight.error_type:
        raise RuntimeError("IDEGym image does not provide the configured Python executable")
    progress(task.task_id, f"Sandbox ready. Executing {len(task.test_cases)} test cases...")
    results = []
    for case in task.test_cases:
        # Only the input goes to the sandbox; the expected answer stays here.
        result = await sandbox.exec(test_command(config.python_executable, case), timeout_s=config.test_timeout_s)
        if result.error_type and result.error_type != "timeout":
            raise RuntimeError(f"IDEGym execution failed: {result.error_type}")
        results.append(result)
    return results


# 3. Grade execution results locally, independently of the model's final message.
def grade_task(body: PythonVerifyRequest, executions: list[SandboxExecResult]) -> PythonVerifyResponse:
    results = []
    for index, (case, execution) in enumerate(zip(body.verifier_metadata.test_cases, executions, strict=True)):
        # Match both JSON value and type: True is not the integer 1 in these tasks.
        passed = (
            execution.return_code == 0
            and execution.error_type is None
            and output_matches(execution.stdout or "", case.expected)
        )
        results.append(
            {
                "index": index,
                "passed": passed,
                "return_code": execution.return_code,
                "error_type": execution.error_type or "",
                "stdout": (execution.stdout or "")[:2000],
                "stderr": (execution.stderr or "")[:2000],
            }
        )
    passed_count = sum(case["passed"] for case in results)
    return PythonVerifyResponse(
        **body.model_dump(),
        reward=float(passed_count == len(results)),
        passed_tests=passed_count,
        total_tests=len(results),
        failure_reason=None if passed_count == len(results) else "test_failure",
        test_results=results,
    )


async def evaluate(body: PythonVerifyRequest, provider: dict, config: PythonResourcesConfig) -> PythonVerifyResponse:
    """Prepare -> run -> grade, called by Gym after SimpleAgent generates an answer."""
    code = extract_code(body.response.output_text or "")
    task = body.verifier_metadata
    if not code:
        return PythonVerifyResponse(
            **body.model_dump(),
            reward=0,
            passed_tests=0,
            total_tests=len(task.test_cases),
            failure_reason="empty_model_output",
        )
    spec = prepare_task(code, task, config)
    progress(task.task_id, "Model response received. Starting an IDEGym sandbox...")
    # AsyncSandbox is NeMo Gym's interface; IDEGym implements provisioning and execution.
    # A fresh sandbox is created for each response; the context owns cleanup.
    async with AsyncSandbox(provider, spec) as sandbox:
        await sandbox.start()
        executions = await run_task(sandbox, task, config)
    progress(task.task_id, "Test execution finished; sandbox closed.")
    return grade_task(body, executions)


async def verify_solution(
    body: PythonVerifyRequest, provider: dict, config: PythonResourcesConfig
) -> PythonVerifyResponse:
    # Gym resources-server callback compatibility; the teaching flow is evaluate().
    return await evaluate(body, provider, config)


# 4. Report scored rollouts; no model parameters are updated in any module.
def report_results(
    results: list[PythonVerifyResponse],
    output: Path,
    *,
    model: str,
    samples_file: Path,
    samples: int,
    repeats: int,
) -> None:
    click.echo("[3/3] Evaluation complete; Gym services stopped.")
    for result in results:
        show_result(result)
    # Since each reward is 0 or 1, mean_reward is the fraction of successful rollouts.
    # This is evaluation only: collecting rewards does not update model parameters.
    summary = {
        "model": model,
        "provider": os.environ.get("MODEL_PROVIDER", "litellm"),
        "inference_host": urlsplit(os.environ["POC_MODEL_BASE_URL"]).hostname,
        "orchestrator_url": os.environ["IDEGYM_ORCHESTRATOR_URL"],
        "rollouts": len(results),
        "samples": samples,
        "samples_file": str(samples_file.resolve()),
        "repeats": repeats,
        "mean_reward": sum(result.reward for result in results) / len(results),
        "parameter_updates": 0,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    click.echo(f"\nMean reward: {summary['mean_reward']:.3f}")
    click.echo(f"Artifacts: {output}")
    click.echo(f"\nTry changing a prompt or test case in {samples_file}, then run again.")


# CLI: configure -> run through NeMo Gym -> report.
@click.command()
@click.option(
    "--samples-file",
    default=SAMPLES_FILE,
    type=click.Path(exists=True, dir_okay=False, resolve_path=True, path_type=Path),
    show_default=True,
    help="JSONL file containing all sample prompts and test cases.",
)
@click.option(
    "--limit", default=1, type=click.IntRange(1), show_default=True, help="Run the first N samples in file order."
)
@click.option("--repeats", default=1, type=click.IntRange(1), show_default=True, help="Model responses per sample.")
@click.option("--test-timeout", default=TEST_TIMEOUT_S, type=click.IntRange(1), show_default=True)
@click.option(
    "--run-timeout",
    default=900,
    type=click.IntRange(1),
    show_default=True,
    help="Maximum seconds for the complete Gym run, including startup and all samples.",
)
@click.option("--model", default=None, help="Override the model selected in .env.")
@click.option("--output-dir", type=click.Path(path_type=Path))
@record_run(MODULE_ROOT)
def main(
    samples_file: Path,
    limit: int,
    repeats: int,
    test_timeout: int,
    run_timeout: int,
    model: str | None,
    output_dir: Path | None,
) -> None:
    """Generate a Python solution with NeMo Gym and verify it in IDEGym."""
    # Read .env and normalize provider URL, API key, and model into POC_MODEL_*.
    # This selects hosted inference; no model weights are loaded on this machine.
    configure(MODULE_ROOT)
    samples = load_samples(samples_file, limit)
    model = model_for_provider(model or os.environ["POC_MODEL_NAME"])
    os.environ["POC_MODEL_NAME"] = model
    output = new_output(MODULE_ROOT, output_dir, "model")
    # Gym reads this snapshot rather than the editable source file, preserving the
    # selected samples even if learners change the source after the run starts.
    dataset = output / "tasks.jsonl"
    dataset.write_text("".join(sample.model_dump_json(exclude_none=True) + "\n" for sample in samples))
    events = output / "progress.jsonl"
    events.touch()
    os.environ["POC_LOG_DIR"] = str(output / "logs")
    # The verifier runs in a separate Gym process. This file relays its workshop
    # progress messages without exposing raw service logs in the terminal.
    os.environ["POC_WORKSHOP_EVENTS"] = str(events)
    configuration = gym_configuration(dataset, test_timeout)
    config_path = output / "config.yaml"
    save_configuration(configuration, output)
    # Reuse the locked project .venv for all three Gym component subprocesses.
    prepare_runtime(
        (
            "resources_servers/idegym_python",
            "responses_api_agents/simple_agent",
            "responses_api_models/inference_provider",
        )
    )

    click.echo("Module 1 — Generate and verify a Python function\n")
    click.echo(f"Samples: {len(samples)} from {samples_file}")
    click.echo("Tasks: " + ", ".join(sample.verifier_metadata.task_id for sample in samples))
    click.echo(f"Model: {model} through {os.environ.get('MODEL_PROVIDER', 'litellm')}")
    click.echo("Interface: NeMo Gym AsyncSandbox backed by IDEGym; no model tools")
    click.echo("Execution: remote IDEGym sandbox")
    click.echo("Grading: fresh sandbox per response; expected answers stay on the client")
    click.echo(f"Evaluation: {repeats} response(s) per sample; no tools or repair loop")
    if len(samples) == 1:
        click.echo("\nModel input:")
        click.echo(samples[0].responses_create_params.model_dump_json(include={"input"}, indent=2))
    # A rollout is one generated response plus its verification result.
    # Repeats create independent rollouts, not extra reasoning steps within one answer.
    expected_rollouts = len(samples) * repeats
    click.echo(f"Run deadline: {run_timeout}s for all samples; status every 30s while waiting")
    click.echo(f"Artifacts and logs: {output}")
    click.echo("\n[1/3] Preparing samples and NeMo Gym evaluation services...")
    click.echo(f"[2/3] Starting local services, then generating and grading {expected_rollouts} solution(s)...")
    rollouts = output / "rollouts.jsonl"
    # Delegate orchestration to the real Gym CLI: it starts the services, asks the
    # agent to generate each answer, invokes verification, and stops the services.
    command = [
        str(Path(sys.executable).with_name("gym")),
        "eval",
        "run",
        "--config",
        str(config_path),
        "--split",
        "validation",
        "--output",
        str(rollouts),
        "--limit",
        str(len(samples)),
        "--num-repeats",
        str(repeats),
        "--concurrency",
        "1",
        "--max-output-tokens",
        str(MAX_OUTPUT_TOKENS),
        "--temperature",
        "0.6",
    ]
    with (output / "gym.log").open("w") as log:
        # Bound the whole orchestration: Gym's internal HTTP retries can otherwise
        # wait indefinitely. run_logged also cleans up this run's process group.
        try:
            returncode = run_with_progress(command, log, events, timeout_s=run_timeout)
        except subprocess.TimeoutExpired as error:
            raise click.ClickException(
                f"Evaluation exceeded the {run_timeout}s run deadline; local Gym processes stopped. "
                f"This is an incomplete evaluation, not reward 0. Logs and partial results: {output}. "
                "Check model/service access before retrying; use --run-timeout to change the deadline."
            ) from error
    # A failed evaluation has no complete score. Do not present it as model reward 0.
    if returncode:
        raise click.ClickException(
            f"Evaluation could not finish (Gym exit {returncode}); this is not reward 0. "
            f"Check service access and the local log: {output / 'gym.log'}"
        )
    # Gym writes scored responses to JSONL. Read those results rather than calling
    # the verifier again, which would provision another set of sandboxes.
    results = [
        PythonVerifyResponse.model_validate_json(line) for line in rollouts.read_text().splitlines() if line.strip()
    ]
    if len(results) != expected_rollouts:
        raise click.ClickException(
            f"Evaluation incomplete: expected {expected_rollouts} results, received {len(results)}"
        )
    report_results(results, output, model=model, samples_file=samples_file, samples=len(samples), repeats=repeats)


if __name__ == "__main__":
    main()
