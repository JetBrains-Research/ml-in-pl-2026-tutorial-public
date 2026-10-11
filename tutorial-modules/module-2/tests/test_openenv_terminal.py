"""Task grading and lifecycle boundaries of the official OpenEnv terminal workshop."""

import json
import shlex
import subprocess
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming

from experiments.ide_mcp.task import Grade
from experiments.ide_mcp.transport import ToolInfo, ToolResult
from experiments.module_3_openenv_bash import transport, workshop
from experiments.module_3_openenv_bash.agent import TerminalRunRequest
from experiments.module_3_openenv_bash.task import (
    CHECKS_MARKER,
    PROJECT_PATH,
    checks_from_output,
    runtime_setup,
    stage_task,
    verifier_setup,
)


@pytest.mark.parametrize("failure", [False, True])
async def test_openenv_server_is_deleted_after_client_session(failure, monkeypatch, tmp_path):
    for key, value in {
        "IDEGYM_ORCHESTRATOR_URL": "https://workshop.example",
        "IDEGYM_AUTH_USERNAME": "test",
        "IDEGYM_AUTH_PASSWORD": "test",
    }.items():
        monkeypatch.setenv(key, value)
    server = SimpleNamespace(server_id=42, openenv_url="https://workshop.example/api/ws-forward/client/42")
    client = SimpleNamespace(start_server=AsyncMock(return_value=server), stop_server=AsyncMock())

    @asynccontextmanager
    async def connection(**kwargs):
        yield client

    @asynccontextmanager
    async def bridge(*args):
        if failure:
            raise ConnectionError("reset failed")
        yield SimpleNamespace(scope={})

    monkeypatch.setattr(transport, "IdeGYMClient", connection)
    monkeypatch.setattr(transport, "OpenEnvBridge", bridge)
    if failure:
        with pytest.raises(ConnectionError):
            async with transport.connect_openenv(
                transport.TerminalSettings(image="registry.example/openenv:provided"), tmp_path
            ):
                pass
    else:
        async with transport.connect_openenv(
            transport.TerminalSettings(image="registry.example/openenv:provided"), tmp_path
        ):
            pass
    client.stop_server.assert_awaited_once_with(
        server, polling_config=client.start_server.call_args.kwargs["polling_config"]
    )
    request = client.start_server.call_args.kwargs
    assert request["server_kind"].value == "openenv"
    assert request["image_tag"] == "registry.example/openenv:provided"
    assert request["container_port"] == 8000
    assert request["run_as_root"]
    assert request["runtime_class_name"] == "gvisor"


@pytest.mark.parametrize("success", [False, True])
async def test_bash_maps_to_native_exec_action(success, tmp_path):
    bridge = transport.OpenEnvBridge("https://workshop.example", tmp_path)
    bridge.request = AsyncMock(
        return_value={
            "observation": {"success": success, "output": "output", "error": "failed"},
            "done": False,
        }
    )
    result = await bridge.call("bash", {"command": "pwd"})
    assert result.is_error == (not success)
    bridge.request.assert_awaited_once_with({"op": "step", "action": {"action_type": "exec", "command": "pwd"}})
    with pytest.raises(ValueError):
        await bridge.call("evaluate", {})
    forbidden = await bridge.call("bash", {"command": "pwd", "action_type": "evaluate"})
    assert forbidden.is_error
    assert bridge.request.await_count == 1


def test_task_contains_fixture_and_trusted_verifier(tmp_path):
    stage_task(tmp_path)
    assert (tmp_path / "project/calculator.py").exists()
    assert "sum_numbers" in (tmp_path / "task/instruction.md").read_text()
    assert "__WORKSHOP_CHECKS__" in (tmp_path / "task/tests/verify.py").read_text()
    assert "reward.txt" in (tmp_path / "task/tests/verify.py").read_text()
    assert "python3 -I" in (tmp_path / "task/tests/test.sh").read_text()


def test_runtime_setup_keeps_verifier_off_server_until_grading(tmp_path):
    project, task = tmp_path / "project", tmp_path / "task"
    setup = runtime_setup(str(project), str(task))
    subprocess.run(shlex.split(setup), check=True)
    assert (project / "calculator.py").exists()
    assert not (task / "tests").exists()
    assert "__WORKSHOP_CHECKS__" not in setup
    subprocess.run(shlex.split(verifier_setup(str(project), str(task))), check=True)
    assert "__WORKSHOP_CHECKS__" in (task / "tests/verify.py").read_text()
    subprocess.run(shlex.split(setup), check=True)
    assert not (task / "tests").exists(), "A new episode must remove the previous verifier"


@pytest.mark.parametrize("output", ["", CHECKS_MARKER + "{}"])
def test_missing_verification_data_is_not_reward_zero(output):
    with pytest.raises((RuntimeError, ValueError)):
        checks_from_output(output)


@pytest.mark.parametrize("golden", [False, True])
@pytest.mark.parametrize("grading_error", [False, True])
async def test_episode_uses_upstream_verifier_reward_and_cleans_up(golden, grading_error, monkeypatch, tmp_path):
    checks = {name: True for name in Grade.model_fields}
    evaluation = {
        "observation": {
            "success": not grading_error,
            "error": "verifier failed" if grading_error else "",
            "output": CHECKS_MARKER + json.dumps(checks),
        },
        "done": not grading_error,
        "reward": None if grading_error else 1.0,
    }
    submission = {"calculator.py": "submitted", "main.py": "main", "test_calculator.py": "tests"}

    async def call(name, arguments):
        text = json.dumps(submission) if "Submission contains a symlink" in arguments["command"] else ""
        return ToolResult(content=[{"type": "text", "text": text}])

    connections = [
        SimpleNamespace(
            scope={"server_id": index + 1},
            discover=AsyncMock(return_value=[ToolInfo(name="bash", input_schema={})]),
            call=AsyncMock(side_effect=call),
            evaluate=AsyncMock(return_value=evaluation),
            state=AsyncMock(return_value={"last_action_type": "evaluate"}),
        )
        for index in range(2)
    ]
    lifecycle = []

    @asynccontextmanager
    async def connect(*args):
        role = "agent" if not lifecycle else "grader"
        selected = connections[0 if role == "agent" else 1]
        lifecycle.append("open-" + role)
        try:
            yield selected
        finally:
            lifecycle.append("close-" + role)

    monkeypatch.setattr(transport, "connect_openenv", connect)
    body = model_call = None
    if not golden:
        body = TerminalRunRequest(responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input="Rename add"))
        model_call = AsyncMock(
            side_effect=[
                NeMoGymResponse(
                    id="response",
                    created_at=0,
                    object="response",
                    model="test",
                    parallel_tool_calls=False,
                    tool_choice="auto",
                    tools=[],
                    status="completed",
                    output=items,
                )
                for items in [
                    [
                        dict(
                            type="function_call",
                            name="bash",
                            arguments='{"command":"rename"}',
                            call_id="call-1",
                            id="item-1",
                        )
                    ],
                    [],
                ]
            ]
        )
    if grading_error:
        with pytest.raises(RuntimeError, match="could not verify"):
            await workshop.evaluate(
                transport.TerminalSettings(image="registry.example/openenv:provided"),
                tmp_path,
                body=body,
                model_call=model_call,
            )
    else:
        result = await workshop.evaluate(
            transport.TerminalSettings(image="registry.example/openenv:provided"),
            tmp_path,
            body=body,
            model_call=model_call,
        )
        assert result.golden == golden and result.reward == result.environment_reward == 1
        assert json.loads((tmp_path / "evaluation.json").read_text()) == evaluation
        if not golden:
            assert result.response.id == "response"
            assert result.responses_create_params == body.responses_create_params
            assert result.successful_tool_calls == 1
    assert lifecycle == ["open-agent", "open-grader", "close-agent", "close-grader"]
    agent, grader = connections
    agent.evaluate.assert_not_awaited()
    grader.evaluate.assert_awaited_once()
    agent_commands = [call.args[1]["command"] for call in agent.call.await_args_list]
    grader_commands = [call.args[1]["command"] for call in grader.call.await_args_list]
    assert any(PROJECT_PATH in command and "json.dumps" in command for command in agent_commands)
    assert verifier_setup() not in agent_commands
    assert grader_commands[-1] == verifier_setup()
    assert "submitted" in grader_commands[0]
    assert json.loads((tmp_path / "final-files.json").read_text()) == submission
