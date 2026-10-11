"""Consistent learner output around Gym's real service and episode lifecycle."""

import asyncio
import json
import os
import subprocess
import sys
from collections.abc import AsyncIterator, Coroutine
from contextlib import (
    AbstractAsyncContextManager,
    asynccontextmanager,
    redirect_stderr,
    redirect_stdout,
)
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generic, TextIO, TypeVar

import click
from pydantic import BaseModel

from experiments.common import progress
from experiments.module_1_python.workshop_support import run_with_progress

Result = TypeVar("Result")
Connection = TypeVar("Connection")


@dataclass
class SandboxPair(Generic[Connection]):
    agent: Connection
    grader: Connection
    _agent_lifetime: asyncio.Task[None] = field(repr=False)
    _agent_stop: asyncio.Event = field(repr=False)

    async def close_agent(self) -> None:
        """End the model sandbox before grading the exported source files."""
        self._agent_stop.set()
        await self._agent_lifetime


@asynccontextmanager
async def concurrent_sandboxes(
    agent_context: AbstractAsyncContextManager[Connection],
    grader_context: AbstractAsyncContextManager[Connection],
) -> AsyncIterator[SandboxPair[Connection]]:
    """Start both sandboxes together and clean up partial starts, errors and cancellation."""
    loop = asyncio.get_running_loop()
    ready: list[asyncio.Future[Connection]] = [loop.create_future(), loop.create_future()]
    stops = [asyncio.Event(), asyncio.Event()]

    async def own_context(
        context: AbstractAsyncContextManager[Connection], connected: asyncio.Future[Connection], stop: asyncio.Event
    ) -> None:
        # MCP/AnyIO cancel scopes must exit in the task that entered them.
        # Keep each owner alive while callers use its connection for the episode.
        try:
            async with context as connection:
                if not connected.done():
                    connected.set_result(connection)
                await stop.wait()
        except BaseException as error:
            if not connected.done():
                connected.set_exception(error)
            raise

    owners = [
        asyncio.create_task(own_context(context, connected, stop))
        for context, connected, stop in zip((agent_context, grader_context), ready, stops, strict=True)
    ]
    try:
        agent, grader = await asyncio.gather(*ready)
        yield SandboxPair(agent, grader, owners[0], stops[0])
    finally:
        for connected, stop, owner in zip(ready, stops, owners, strict=True):
            if connected.done() and not connected.cancelled() and connected.exception() is None:
                stop.set()
            elif not owner.done():
                owner.cancel()
        # Join owners before returning, including partially completed startups.
        closed = await asyncio.gather(*owners, return_exceptions=True)
        for connected in ready:
            if connected.done() and not connected.cancelled():
                connected.exception()
        for error in closed:
            if isinstance(error, Exception):
                raise error


async def run_concurrently(
    first: Coroutine[Any, Any, Result], second: Coroutine[Any, Any, Result]
) -> tuple[Result, Result]:
    """Run paired startup/preparation operations, joining cancelled work before cleanup."""
    tasks = [asyncio.create_task(first), asyncio.create_task(second)]
    try:
        first_result, second_result = await asyncio.gather(*tasks)
        return first_result, second_result
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        # Join cancelled preparation before its sandbox connection closes.
        await asyncio.gather(*tasks, return_exceptions=True)


def event(message: str) -> None:
    """Relay safe stage descriptions from a Gym worker; keep payloads in artifacts."""
    progress("episode", message)


def announce(
    number: int, title: str, task: str, interface: str, model: str, golden: bool, step_limit: int, output: Path
) -> None:
    click.echo(f"Module {number} — {title}\n")
    click.echo(f"Task: {task}")
    click.echo(f"Interface: {interface}")
    click.echo(
        "Model: reference transformation (no inference)"
        if golden
        else f"Model: {model} through {os.environ.get('MODEL_PROVIDER', 'litellm')}"
    )
    click.echo(f"Evaluation: {'reference check' if golden else f'up to {step_limit} model calls'}")
    click.echo(f"Artifacts and logs: {output}")
    click.echo("[1/3] Preparing the remote task environment and evaluation services...")


def run_gym(command: list[str], log: TextIO, output: Path, *, timeout_s: int = 1800) -> int:
    events = output / "progress.jsonl"
    events.touch()
    previous_events = os.environ.get("POC_WORKSHOP_EVENTS")
    os.environ["POC_WORKSHOP_EVENTS"] = str(events)
    click.echo(f"[2/3] Starting local services, then the episode and grading; run deadline {timeout_s}s.")
    try:
        return run_with_progress(command, log, events, timeout_s=timeout_s)
    except subprocess.TimeoutExpired as error:
        raise click.ClickException(
            f"Evaluation exceeded {timeout_s}s; local Gym processes stopped. "
            f"Incomplete evaluation, not reward 0. Inspect {output / 'gym.log'}."
        ) from error
    finally:
        if previous_events is None:
            os.environ.pop("POC_WORKSHOP_EVENTS", None)
        else:
            os.environ["POC_WORKSHOP_EVENTS"] = previous_events


def run_reference(episode: Coroutine[Any, Any, Result], timeout_s: float, output: Path) -> Result:
    """Bound the reference check while allowing sandbox context managers to clean up."""
    console = sys.stdout
    events = output / "progress.jsonl"
    events.touch()
    previous_events = os.environ.get("POC_WORKSHOP_EVENTS")
    os.environ["POC_WORKSHOP_EVENTS"] = str(events)
    stopped = asyncio.Event()

    async def heartbeat() -> None:
        with events.open() as stream:
            last_message = asyncio.get_running_loop().time()
            while True:
                for line in stream:
                    entry = json.loads(line)
                    click.echo(f"[2/3] {entry['task_id']}: {entry['message']}", file=console)
                    last_message = asyncio.get_running_loop().time()
                if stopped.is_set():
                    return
                if asyncio.get_running_loop().time() - last_message >= 30:
                    click.echo("Still running: waiting for the reference episode or grading to finish.", file=console)
                    last_message = asyncio.get_running_loop().time()
                await asyncio.sleep(0.2)

    async def bounded() -> Result:
        status = asyncio.create_task(heartbeat())
        try:
            async with asyncio.timeout(timeout_s):
                return await episode
        finally:
            stopped.set()
            await status

    click.echo(f"[2/3] Applying the reference transformation and grading; deadline {timeout_s}s.")
    try:
        # Reference checks call SDKs in this process rather than through Gym.
        # Keep their raw output private, just like the model run's gym.log.
        with (output / "reference.log").open("w") as log, redirect_stdout(log), redirect_stderr(log):
            return asyncio.run(bounded())
    except TimeoutError as error:
        raise click.ClickException(
            f"Reference evaluation exceeded {timeout_s}s; incomplete evaluation, not reward 0. Artifacts: {output}"
        ) from error
    finally:
        if previous_events is None:
            os.environ.pop("POC_WORKSHOP_EVENTS", None)
        else:
            os.environ["POC_WORKSHOP_EVENTS"] = previous_events


def show_checks(checks: BaseModel, reward: float, output: Path) -> None:
    click.echo("[3/3] Evaluation complete; remote sandbox closed.")
    for name, passed in checks.model_dump().items():
        click.echo(f"  {'PASS' if passed else 'FAIL'}  {name.replace('_', ' ')}")
    click.echo(f"Reward: {reward:.1f}/1")
    click.echo(f"Artifacts: {output}")
