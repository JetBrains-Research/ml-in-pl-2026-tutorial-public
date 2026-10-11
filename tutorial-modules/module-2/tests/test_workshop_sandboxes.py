"""Concurrency and cleanup at the independent agent/grader sandbox boundary."""

import asyncio
from contextlib import asynccontextmanager

import pytest

from experiments.workshop_support import concurrent_sandboxes


async def test_both_sandboxes_start_together_and_agent_closes_before_grading():
    started, closed = [], []
    both_started = asyncio.Event()

    @asynccontextmanager
    async def connect(role):
        owner = asyncio.current_task()
        started.append(role)
        if len(started) == 2:
            both_started.set()
        # Sequential startup would deadlock here.
        await both_started.wait()
        try:
            yield role
        finally:
            assert asyncio.current_task() is owner, "MCP connections must close in their owning task"
            closed.append(role)

    async with asyncio.timeout(5):
        async with concurrent_sandboxes(connect("agent"), connect("grader")) as sandboxes:
            assert sandboxes.agent == "agent" and sandboxes.grader == "grader"
            await sandboxes.close_agent()
            assert closed == ["agent"]
    assert closed == ["agent", "grader"]


@pytest.mark.parametrize("failed_role", ["agent", "grader"])
async def test_partial_startup_failure_closes_successful_peer(failed_role):
    started, closed = [], []
    both_started = asyncio.Event()

    @asynccontextmanager
    async def connect(role):
        started.append(role)
        if len(started) == 2:
            both_started.set()
        try:
            await both_started.wait()
            if role == failed_role:
                raise RuntimeError("image failed to start")
            yield role
        finally:
            closed.append(role)

    with pytest.raises(RuntimeError, match="image failed to start"):
        async with concurrent_sandboxes(connect("agent"), connect("grader")):
            pytest.fail("A partially started pair cannot run an episode")
    assert sorted(closed) == ["agent", "grader"]


async def test_cancelled_startup_joins_both_contexts_before_returning():
    started, closed = [], []
    both_started = asyncio.Event()

    @asynccontextmanager
    async def connect(role):
        started.append(role)
        if len(started) == 2:
            both_started.set()
        try:
            await asyncio.Event().wait()
            yield role
        finally:
            closed.append(role)

    async def episode():
        async with concurrent_sandboxes(connect("agent"), connect("grader")):
            pytest.fail("Startup was cancelled before either sandbox became ready")

    task = asyncio.create_task(episode())
    await asyncio.wait_for(both_started.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert sorted(closed) == ["agent", "grader"]
