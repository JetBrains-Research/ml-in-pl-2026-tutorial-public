"""Exercise the real teaching entry points at the hosted-inference boundary."""

import importlib
import json
from contextlib import asynccontextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from click.testing import CliRunner
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming

from experiments.ide_mcp.task import Grade
from experiments.ide_mcp.transport import ToolInfo, ToolResult

MODULES = (
    "module_2_direct_bash",
    "module_3_openenv_bash",
    "module_4_jetbrains_mcp",
)


@pytest.mark.parametrize("module", MODULES)
@pytest.mark.parametrize("failure", [False, True])
def test_workshop_cli_reports_checks_or_incomplete_run(module, failure, monkeypatch, tmp_path):
    workshop = importlib.import_module(f"experiments.{module}.workshop")
    owner = workshop
    if module == "module_3_openenv_bash":
        listener = SimpleNamespace(bind=lambda *_: None, getsockname=lambda: ("127.0.0.1", 12345))
        monkeypatch.setattr(workshop.socket, "socket", lambda: nullcontext(listener))
    monkeypatch.setattr("experiments.common.configure", lambda *_: None)
    for target in {workshop, owner}:
        if hasattr(target, "configure"):
            monkeypatch.setattr(target, "configure", lambda *_: None)
    monkeypatch.setattr(
        "experiments.common.prepare_runtime"
        if module == "module_3_openenv_bash"
        else f"{owner.__name__}.prepare_runtime",
        lambda *_: None,
    )
    monkeypatch.setenv("POC_MODEL_NAME", "vertex_ai/google/gemma-4-26b-a4b-it-maas")
    monkeypatch.setenv("MODEL_PROVIDER", "litellm")
    monkeypatch.setenv("IDEGYM_AUTH_PASSWORD", "never-print-this-password")
    output = tmp_path / "output"

    def gym(command, log, directory, *, timeout_s):
        assert timeout_s == 42
        if failure:
            return 1
        report = {
            "reward": 1,
            "checks": {name: True for name in Grade.model_fields},
            "mode": "openenv"
            if module == "module_3_openenv_bash"
            else "bash"
            if module == "module_2_direct_bash"
            else "bundled",
            "golden": False,
            "tool_calls": 1,
            "successful_tool_calls": 1,
        }
        if module == "module_3_openenv_bash":
            report["environment_reward"] = 1.0
        if module == "module_4_jetbrains_mcp":
            report = report | {
                "checks": dict(
                    baseline_detected=True,
                    inspections_clean=True,
                    behavior_preserved=True,
                    model_used_inspections=True,
                ),
                "before_counts": {"pricing.py": 1, "receipt.py": 1},
                "after_counts": {"pricing.py": 0, "receipt.py": 0},
                "behavior": {"pricing": True, "receipt": True},
            }
            (directory / "verification.json").write_text(json.dumps(report))
        (directory / "rollouts.jsonl").write_text(json.dumps(report) + "\n")
        return 0

    monkeypatch.setattr(
        "experiments.workshop_support.run_gym" if module == "module_3_openenv_bash" else f"{owner.__name__}.run_gym",
        gym,
    )
    result = CliRunner().invoke(workshop.main, ["--output-dir", str(output), "--run-timeout", "42"])
    assert "Module " in result.output and "Interface:" in result.output
    assert str(output) in result.output
    assert "never-print-this-password" not in result.output
    if failure:
        assert result.exit_code == 1
        assert "[3/3]" not in result.output
        assert not (output / "summary.json").exists()
    else:
        assert result.exit_code == 0, result.output
        assert "[3/3]" in result.output
        summary = json.loads((output / "summary.json").read_text())
        assert summary["reward"] == 1
        assert summary["model"] == "vertex_ai/google/gemma-4-26b-a4b-it-maas"


@pytest.mark.parametrize("module", ["module_2_direct_bash"])
@pytest.mark.parametrize("uses_tool,grader_fails", [(True, False), (False, False), (True, True)])
async def test_central_episode_uses_real_loop_and_independent_grader(
    module, uses_tool, grader_fails, monkeypatch, tmp_path
):
    workshop = importlib.import_module(f"experiments.{module}.workshop")
    name = "bash"
    arguments = {"command": "rename"}
    checks = Grade(**{key: True for key in Grade.model_fields})
    submission = {"calculator.py": "submitted", "main.py": "main", "test_calculator.py": "tests"}
    agent_execute = AsyncMock(return_value=SimpleNamespace(exit_code=0, stdout=json.dumps(submission)))
    grader_execute = AsyncMock(
        side_effect=[
            SimpleNamespace(exit_code=0),
            SimpleNamespace(exit_code=int(grader_fails), stdout=checks.model_dump_json()),
        ]
    )
    connections = [
        SimpleNamespace(
            discover=AsyncMock(return_value=[ToolInfo(name=name, input_schema={})]),
            prepare=AsyncMock(),
            call=AsyncMock(return_value=ToolResult(content=[])),
            execute_bash=execute,
            server=SimpleNamespace(server_id=index + 1, execute_bash=execute),
        )
        for index, execute in enumerate((agent_execute, grader_execute))
    ]
    connection = connections[0]
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

    monkeypatch.setattr(workshop, "connect_bash", connect)
    monkeypatch.delenv("POC_WORKSHOP_EVENTS", raising=False)
    outputs = []
    if uses_tool:
        outputs.append(
            [dict(type="function_call", name=name, arguments=json.dumps(arguments), call_id="call-1", id="item-1")]
        )
    outputs.append([])
    model_call = AsyncMock(
        side_effect=[
            NeMoGymResponse(
                id="response",
                created_at=0,
                object="response",
                model="test",
                output=items,
                parallel_tool_calls=False,
                tool_choice="auto",
                tools=[],
                status="completed",
            )
            for items in outputs
        ]
    )
    agent = importlib.import_module("experiments.bash_access.agent")
    request_type = agent.BashRunRequest
    request = request_type(responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input="Rename add"))
    settings = workshop.BashSettings(mode="bash", image="registry/image:tag")
    episode = workshop.evaluate_bash
    if grader_fails:
        with pytest.raises(RuntimeError, match="verifier failed"):
            await episode(request, settings, model_call, 3, tmp_path)
    else:
        result = await episode(request, settings, model_call, 3, tmp_path)
        assert result.reward == float(uses_tool)
        assert result.successful_tool_calls == int(uses_tool)
        assert json.loads((tmp_path / "final-files.json").read_text())["calculator.py"] == "submitted"
    assert lifecycle == ["open-agent", "open-grader", "close-agent", "close-grader"]
    for prepared in connections:
        prepared.prepare.assert_awaited_once()
    identities = json.loads((tmp_path / "sandboxes.json").read_text())
    assert identities["agent"]["server_id"] != identities["grader"]["server_id"]
    assert "renamed" not in agent_execute.call_args.args[0], "Agent must not execute the verifier"
    assert "renamed" in grader_execute.call_args.args[0]
    payload = model_call.call_args_list[0].args[0]
    assert [tool["name"] for tool in payload.tools] == [name]
    if uses_tool:
        connection.call.assert_awaited_once_with(name, arguments)
        assert model_call.call_args_list[1].args[0].input[-1].type == "function_call_output"
    assert (tmp_path / "trajectory.jsonl").exists()


def test_reference_deadline_cleans_up_episode(tmp_path):
    import asyncio

    import click

    from experiments.workshop_support import run_reference

    lifecycle = []

    async def episode():
        try:
            await asyncio.sleep(60)
        finally:
            lifecycle.append("closed")

    with pytest.raises(click.ClickException, match="incomplete evaluation"):
        run_reference(episode(), 0.01, tmp_path)
    assert lifecycle == ["closed"]


def test_reference_keeps_sdk_output_in_private_log(tmp_path, capsys):
    from experiments.workshop_support import run_reference

    async def reference():
        print("raw SDK output")
        return "graded"

    assert run_reference(reference(), 1, tmp_path) == "graded"
    assert "raw SDK output" not in capsys.readouterr().out
    assert "raw SDK output" in (tmp_path / "reference.log").read_text()
