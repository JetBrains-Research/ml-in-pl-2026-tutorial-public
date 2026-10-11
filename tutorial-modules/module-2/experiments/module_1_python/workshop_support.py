# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Data types, output formatting, and process plumbing for the workshop."""

import json
import re
import shlex
from pathlib import Path
from threading import Event, Thread
from time import monotonic
from typing import ClassVar, TextIO

import click
from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
)
from pydantic import BaseModel, Field

from experiments.common import progress as progress
from experiments.common import run_logged


class TestCase(BaseModel):
    input: int | str | list[int]
    expected: int | bool | list[int]


class TaskMetadata(BaseModel):
    task_id: str
    test_cases: list[TestCase] = Field(min_length=1)


class PythonSample(BaseRunRequest):
    verifier_metadata: TaskMetadata


class PythonVerifyRequest(BaseVerifyRequest):
    verifier_metadata: TaskMetadata


class PythonVerifyResponse(BaseVerifyResponse):
    verifier_metadata: TaskMetadata
    passed_tests: int
    total_tests: int
    failure_reason: str | None = None
    test_results: list[dict[str, str | int | bool]] = Field(default_factory=list)


class PythonResourcesConfig(BaseResourcesServerConfig):
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS
    sandbox_provider: str = "sandbox"
    image: str
    python_executable: str = "/opt/idegym/.venv/bin/python"
    test_timeout_s: int = Field(default=5, ge=1)
    ready_timeout_s: int = Field(default=300, ge=1)
    concurrency: int = Field(default=1, ge=1)


def extract_code(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    fences = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.DOTALL)
    return (fences[-1] if fences else text).strip()


def test_command(python_executable: str, case: TestCase) -> str:
    """Pass inputs to the sandbox; keep expected answers in the verifier."""
    runner = (
        "import json, runpy, sys; "
        "solve = runpy.run_path('/tmp/solution.py')['solve']; "
        "print(json.dumps(solve(json.loads(sys.argv[1]))))"
    )
    return shlex.join([python_executable, "-I", "-c", runner, json.dumps(case.input)])


def output_matches(output: str, expected: int | bool | list[int]) -> bool:
    try:
        actual = json.loads(output)
    except (ValueError, TypeError):
        return False
    # Python considers True == 1; these tasks require the JSON type to match too.
    return type(actual) is type(expected) and actual == expected


def run_with_progress(
    command: list[str], log: TextIO, events: Path, *, timeout_s: float | None = None, heartbeat_s: float = 30
) -> int:
    stopped = Event()

    def relay() -> None:
        last_message = monotonic()
        stage = "Waiting for local Ray startup"
        with events.open() as stream:
            while True:
                line = stream.readline()
                if line:
                    event = json.loads(line)
                    if event["task_id"] == "startup":
                        click.echo(f"[startup] {event['message']}")
                    else:
                        click.echo(f"[2/3] {event['task_id']}: {event['message']}")
                    stage = f"Waiting after {event['task_id']}: {event['message']}"
                    last_message = monotonic()
                elif stopped.is_set():
                    break
                else:
                    if monotonic() - last_message >= heartbeat_s:
                        click.echo(f"Still running: {stage}. No new progress event in {heartbeat_s:g}s.")
                        last_message = monotonic()
                    stopped.wait(0.2)

    thread = Thread(target=relay, daemon=True)
    thread.start()
    try:
        return run_logged(command, log, timeout_s=timeout_s)
    finally:
        stopped.set()
        thread.join()


def show_result(result: PythonVerifyResponse) -> None:
    click.echo(f"\nTask: {result.verifier_metadata.task_id}\n\nGenerated solution:")
    code = extract_code(result.response.output_text or "")
    for line in (code or "(empty response)").splitlines():
        click.echo("    " + line)
    click.echo("\nTest cases:")
    if not result.test_results:
        click.echo("  No cases executed: the model returned no Python source.")
    for case, checked in zip(result.verifier_metadata.test_cases, result.test_results, strict=False):
        status = "PASS" if checked["passed"] else "FAIL"
        actual = str(checked["stdout"]).strip() or "(no output)"
        click.echo(f"  {status}  solve({case.input!r}) -> {actual}; expected {json.dumps(case.expected)}")
        if checked["error_type"]:
            click.echo(f"        Execution error: {checked['error_type']}")
        if checked["stderr"]:
            click.echo("        " + str(checked["stderr"]).strip().replace("\n", "\n        "))
    click.echo(f"\nReward: {result.reward:.1f} — {result.passed_tests}/{result.total_tests} cases passed")
