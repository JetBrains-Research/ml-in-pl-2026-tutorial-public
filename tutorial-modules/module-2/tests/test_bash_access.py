import asyncio
import json
import shlex
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from idegym.api.tools.bash import BashCommandResponse
from omegaconf import OmegaConf

from experiments.bash_access.agent import BashAgentConfig, BashRunRequest
from experiments.bash_access.cli import BashEvaluationResult
from experiments.bash_access.transport import BashConnection, wait_for_forwarding
from experiments.ide_mcp.task import Grade

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("status", [502, 503, 504, 401])
async def test_forwarding_wait_retries_only_gateway_readiness_errors(status, monkeypatch):
    # Match the SDK's RuntimeError wrapping an HTTPStatusError.
    try:
        raise httpx.HTTPStatusError(
            "gateway not ready",
            request=httpx.Request("GET", "https://orchestrator.example/health"),
            response=httpx.Response(status),
        )
    except httpx.HTTPStatusError:
        try:
            raise RuntimeError("Forwarding failed")
        except RuntimeError as error:
            failure = error
    monkeypatch.setattr("experiments.bash_access.transport.asyncio.sleep", AsyncMock())
    server = SimpleNamespace(forward=AsyncMock(side_effect=[failure, {"status": "ok"}]))
    if status == 401:
        with pytest.raises(RuntimeError, match="Forwarding failed"):
            await wait_for_forwarding(server)
        assert server.forward.await_count == 1
    else:
        await wait_for_forwarding(server)
        assert server.forward.await_count == 2
    server.forward.assert_awaited_with("GET", "health", request_timeout=10)


async def test_forwarding_readiness_has_its_own_deadline():
    async def stalled(*args, **kwargs):
        await asyncio.Event().wait()

    server = SimpleNamespace(forward=AsyncMock(side_effect=stalled))
    with pytest.raises(RuntimeError, match="forwarding readiness"):
        await wait_for_forwarding(server, timeout=0.01)


async def test_direct_bash_uses_http_api_and_preserves_shell_syntax() -> None:
    server = SimpleNamespace(
        execute_bash=AsyncMock(return_value=BashCommandResponse(stdout="yes", stderr="", exit_code=0))
    )
    connection = BashConnection(server)
    command = "printf '%s\\n' yes | cat > output.txt; cat output.txt"
    result = await connection.call("bash", {"command": command, "timeout": 12})
    assert not result.is_error
    invocation = server.execute_bash.call_args.kwargs
    assert invocation["command_timeout"] == 12
    assert invocation["max_output_bytes"] == 24000
    assert shlex.split(invocation["script"])[-1] == command
    assert "cd /home/appuser/work" in invocation["script"]
    assert "PATH=/opt/idegym/.venv/bin" in invocation["script"]


@pytest.mark.parametrize(
    "arguments", [{"command": "ls", "timeout": 121}, {"command": ""}, {"command": "ls", "cwd": "/"}]
)
async def test_invalid_bash_calls_do_not_reach_environment(arguments: dict) -> None:
    server = SimpleNamespace(execute_bash=AsyncMock())
    result = await BashConnection(server).call("bash", arguments)
    assert result.is_error
    server.execute_bash.assert_not_awaited()


async def test_transport_failure_aborts_evaluation() -> None:
    server = SimpleNamespace(execute_bash=AsyncMock(side_effect=ConnectionError("socket closed")))
    with pytest.raises(ConnectionError):
        await BashConnection(server).call("bash", {"command": "ls"})


async def test_direct_bash_exposes_only_the_shell_tool() -> None:
    tools = await BashConnection(None).discover()
    assert [tool.name for tool in tools] == ["bash"]
    assert set(tools[0].input_schema["properties"]) == {"command", "timeout"}


def test_direct_bash_and_openenv_modules_keep_the_same_task(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BASH_COMPARISON_IMAGE", "registry/image:tag")
    path = ROOT / "experiments/module_2_direct_bash"
    config = OmegaConf.to_container(
        OmegaConf.load(path / "config.yaml").bash_agent.responses_api_agents.bash_agent, resolve=True
    )
    config.update(host="127.0.0.1", port=9000, name="bash_agent")
    from experiments.images import BOOKWORM_IMAGE

    parsed = BashAgentConfig.model_validate(config)
    assert parsed.shell.mode == "bash"
    assert parsed.shell.image == BOOKWORM_IMAGE
    request = BashRunRequest.model_validate_json((path / "data/task.jsonl").read_text())
    ide_task = json.loads((ROOT / "experiments/module_3_openenv_bash/data/task.jsonl").read_text())
    assert (
        request.responses_create_params.input[1].content == ide_task["responses_create_params"]["input"][1]["content"]
    )


async def test_direct_bash_stages_task_and_dependencies_before_verification() -> None:
    from experiments.calculator import FIXTURE

    checks = Grade(renamed=False, references_updated=False, arithmetic=False, zero_division=True, main_output=True)
    server = SimpleNamespace(
        execute_bash=AsyncMock(
            side_effect=[
                BashCommandResponse(stdout="", stderr="", exit_code=0),
                BashCommandResponse(stdout="", stderr="", exit_code=0),
                BashCommandResponse(stdout=checks.model_dump_json(), stderr="", exit_code=0),
            ]
        )
    )
    await BashConnection(server).prepare([])
    calls = server.execute_bash.call_args_list
    assert all(name in calls[0].kwargs["script"] for name in FIXTURE)
    assert "pytest==8.4.2" in calls[1].kwargs["script"]
    assert "renamed" in calls[2].kwargs["script"]


def test_shell_reward_requires_successful_interaction() -> None:
    checks = Grade(renamed=True, references_updated=True, arithmetic=True, zero_division=True, main_output=True)
    with pytest.raises(ValueError, match="Reward disagrees"):
        BashEvaluationResult(reward=1, checks=checks, mode="bash", tool_calls=0, successful_tool_calls=0)
    assert (
        BashEvaluationResult(
            reward=1, checks=checks, mode="bash", golden=True, tool_calls=0, successful_tool_calls=0
        ).reward
        == 1
    )


async def test_direct_bash_sandbox_cleanup_on_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from contextlib import asynccontextmanager

    from experiments.bash_access import transport

    for key, value in {
        "IDEGYM_ORCHESTRATOR_URL": "https://example.test",
        "IDEGYM_AUTH_USERNAME": "test-user",
        "IDEGYM_AUTH_PASSWORD": "test-password",
    }.items():
        monkeypatch.setenv(key, value)
    server = SimpleNamespace(forward=AsyncMock(return_value={"status": "ok"}))
    client = SimpleNamespace(start_server=AsyncMock(return_value=server), stop_server=AsyncMock())

    @asynccontextmanager
    async def fake_client(**kwargs):
        yield client

    monkeypatch.setattr(transport, "IdeGYMClient", fake_client)
    with pytest.raises(ConnectionError):
        async with transport.connect_bash(transport.BashSettings(mode="bash", image="registry/image:tag"), tmp_path):
            raise ConnectionError("evaluation failed")
    client.stop_server.assert_awaited_once()
    options = client.start_server.call_args.kwargs
    assert options["run_as_root"] is False
    assert options["reuse_strategy"].value == "NONE"
    assert options["container_port"] == 8000
    assert options["server_kind"].value == "idegym"
