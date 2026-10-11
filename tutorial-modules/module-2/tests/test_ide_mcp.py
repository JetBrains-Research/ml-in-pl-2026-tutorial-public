import subprocess
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import yaml
from mcp.types import CallToolResult, TextContent
from nemo_gym.base_resources_server import BaseRunRequest
from nemo_gym.global_config import ATTEMPT_INDEX_KEY_NAME, ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME
from nemo_gym.openai_utils import NeMoGymChatCompletion, NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.responses_converter import ResponsesConverter
from omegaconf import OmegaConf
from pydantic import ValidationError

from experiments.bash_access.agent import BashRunRequest
from experiments.ide_mcp.agent import IdeAgentConfig, model_tool_output, run_episode
from experiments.ide_mcp.cli import check_image
from experiments.ide_mcp.task import GOLD_SCRIPT, PROJECT_PATH, VERIFY_SCRIPT, parse_grade
from experiments.ide_mcp.transport import IdeConnection, IdeSettings, ToolInfo, ToolResult, select_tools, unpack_result
from experiments.images import PYCHARM_IMAGE
from experiments.module_3_openenv_bash.agent import TerminalAgent, TerminalRunRequest
from experiments.module_4_jetbrains_mcp.agent import InspectionRunRequest

ROOT = Path(__file__).resolve().parents[1]


def test_catalog_excludes_steroid_and_gateway_tools() -> None:
    tools = [
        ToolInfo(name=name, input_schema={})
        for name in (
            "create_file",
            "edit_file",
            "openhands_bash",
            "pycharm_rename_refactoring",
            "pycharm_steroid_execute_code",
        )
    ]
    assert [tool.name for tool in select_tools(tools, "bundled")] == ["pycharm_rename_refactoring"]


def test_mcp_structured_and_text_results_and_errors() -> None:
    assert unpack_result(CallToolResult(content=[], structuredContent={"tools": []})) == {"tools": []}
    assert unpack_result(CallToolResult(content=[TextContent(type="text", text='{"tools": []}')])) == {"tools": []}
    with pytest.raises(RuntimeError, match="Orchestrator MCP tool failed"):
        unpack_result(CallToolResult(content=[], isError=True))


async def test_tool_call_preserves_scope_schema_and_inner_error() -> None:
    session = SimpleNamespace(
        call_tool=AsyncMock(
            return_value=CallToolResult(content=[], structuredContent={"content": [], "is_error": True})
        )
    )
    connection = IdeConnection(
        session, SimpleNamespace(server_id=42), "client-id", IdeSettings(mode="bundled", image="registry/image:tag")
    )
    result = await connection.call("pycharm_rename_refactoring", {"symbolName": "add"})
    assert result.is_error
    assert session.call_tool.call_args.args == (
        "call_server_mcp_tool",
        {
            "request": {
                "client_id": "client-id",
                "server_id": 42,
                "tool_name": "pycharm_rename_refactoring",
                "arguments": {"symbolName": "add"},
            }
        },
    )


async def test_gateway_wrapped_ide_error_is_returned_to_model() -> None:
    session = SimpleNamespace(
        call_tool=AsyncMock(
            return_value=CallToolResult(content=[TextContent(type="text", text="Specify projectPath")], isError=True)
        )
    )
    connection = IdeConnection(
        session, SimpleNamespace(server_id=42), "client-id", IdeSettings(mode="bundled", image="registry/image:tag")
    )
    result = await connection.call("pycharm_get_file_text_by_path", {})
    assert result.is_error
    assert result.content[0]["text"] == "Specify projectPath"


def test_screenshots_remain_images() -> None:
    result = ToolResult(content=[{"type": "image", "mimeType": "image/png", "data": "YWJj"}])
    blocks = model_tool_output(result)
    assert blocks[1] == {"type": "input_image", "image_url": "data:image/png;base64,YWJj", "detail": "auto"}


def test_uppercase_image_tags_fail_before_starting() -> None:
    with pytest.raises(ValidationError):
        IdeSettings(mode="bundled", image="registry.example.com/ide:20260930T120625Z")


def response(output: list[dict]) -> NeMoGymResponse:
    return NeMoGymResponse(
        id="response-1",
        created_at=0,
        object="response",
        model="test",
        output=output,
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
        status="completed",
    )


async def test_model_loop_uses_discovered_schema_and_returns_tool_feedback(tmp_path: Path) -> None:
    call = {
        "type": "function_call",
        "name": "pycharm_rename_refactoring",
        "arguments": '{"symbolName":"add"}',
        "call_id": "call-1",
        "id": "item-1",
    }
    final = {
        "type": "message",
        "role": "assistant",
        "id": "item-2",
        "status": "completed",
        "content": [{"type": "output_text", "text": "Done", "annotations": []}],
    }
    model = AsyncMock(side_effect=[response([call]), response([final])])
    connection = SimpleNamespace(
        call=AsyncMock(return_value=ToolResult(content=[{"type": "text", "text": "Renamed"}]))
    )
    tool = ToolInfo(
        name=call["name"], input_schema={"type": "object", "properties": {"symbolName": {"type": "string"}}}
    )
    result, attempts, successes = await run_episode(
        NeMoGymResponseCreateParamsNonStreaming(input="Rename add"),
        [tool],
        connection,
        model,
        3,
        tmp_path / "trace.jsonl",
    )
    assert (attempts, successes) == (1, 1)
    connection.call.assert_awaited_once_with(call["name"], {"symbolName": "add"})
    # Gym's inference adapter serializes with exclude_unset, so the discriminator must be explicit.
    second_request = model.call_args_list[1].args[0].model_dump(exclude_unset=True)
    assert second_request["input"][-1]["type"] == "function_call_output"
    first = model.call_args_list[0].args[0].model_dump()
    assert first["tools"][0]["parameters"] == tool.input_schema
    second = model.call_args_list[1].args[0]
    assert second.input[-1].type == "function_call_output"
    assert result.output[-1].type == "message"
    assert "Renamed" in (tmp_path / "trace.jsonl").read_text()


@pytest.mark.parametrize(
    "name,arguments", [("run_bash_command", "{}"), ("pycharm_read_file", "[]"), ("pycharm_read_file", "invalid")]
)
async def test_invalid_calls_never_reach_gateway(name: str, arguments: str, tmp_path: Path) -> None:
    model = AsyncMock(
        return_value=response(
            [{"type": "function_call", "name": name, "arguments": arguments, "call_id": "c", "id": "i"}]
        )
    )
    connection = SimpleNamespace(call=AsyncMock())
    result, attempts, successes = await run_episode(
        NeMoGymResponseCreateParamsNonStreaming(input="task"),
        [ToolInfo(name="pycharm_read_file", input_schema={})],
        connection,
        model,
        1,
        tmp_path / "trace.jsonl",
    )
    assert (attempts, successes) == (1, 0)
    assert result.status == "incomplete"
    connection.call.assert_not_awaited()


async def test_transport_failures_abort_instead_of_becoming_model_scores(tmp_path: Path) -> None:
    model = AsyncMock(
        return_value=response(
            [{"type": "function_call", "name": "pycharm_read_file", "arguments": "{}", "call_id": "c", "id": "i"}]
        )
    )
    connection = SimpleNamespace(call=AsyncMock(side_effect=TimeoutError("gateway unavailable")))
    with pytest.raises(TimeoutError):
        await run_episode(
            NeMoGymResponseCreateParamsNonStreaming(input="task"),
            [ToolInfo(name="pycharm_read_file", input_schema={})],
            connection,
            model,
            1,
            tmp_path / "trace.jsonl",
        )


@pytest.fixture
def calculator(tmp_path: Path) -> Path:
    # Minimal behavior-equivalent fixture, so tests don't require the IdeGYM checkout.
    (tmp_path / "calculator.py").write_text(
        "def add(a, b): return a + b\ndef subtract(a, b): return a - b\n"
        "def multiply(a, b): return a * b\ndef divide(a, b):\n"
        '    if b == 0: raise ValueError("Cannot divide by zero")\n    return a / b\n'
    )
    (tmp_path / "main.py").write_text(
        "from calculator import add, subtract, multiply, divide\ndef main():\n"
        '    print(f"2 + 3 = {add(2, 3)}")\n    print(f"10 - 4 = {subtract(10, 4)}")\n'
        '    print(f"6 * 7 = {multiply(6, 7)}")\n    print(f"15 / 3 = {divide(15, 3)}")\n'
    )
    (tmp_path / "test_calculator.py").write_text("from calculator import add\ndef test_add(): assert add(2, 3) == 5\n")
    return tmp_path


def run_script(source: str, project: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-I", "-c", source.replace(PROJECT_PATH, str(project))],
        capture_output=True,
        text=True,
        check=True,
    )


def test_baseline_fails_gold_passes_and_broken_behavior_fails(calculator: Path) -> None:
    baseline = parse_grade(run_script(VERIFY_SCRIPT, calculator).stdout, 0)
    assert baseline.reward == 0
    assert baseline.zero_division
    run_script(GOLD_SCRIPT, calculator)
    assert parse_grade(run_script(VERIFY_SCRIPT, calculator).stdout, 0).reward == 1
    path = calculator / "calculator.py"
    path.write_text(path.read_text().replace("return a + b", "return 5"))
    broken = parse_grade(run_script(VERIFY_SCRIPT, calculator).stdout, 0)
    assert broken.reward == 0
    assert broken.renamed and not broken.arithmetic


def test_missing_test_references_and_malformed_verifier_report_fail(calculator: Path) -> None:
    run_script(GOLD_SCRIPT, calculator)
    (calculator / "test_calculator.py").write_text("")
    assert not parse_grade(run_script(VERIFY_SCRIPT, calculator).stdout, 0).references_updated
    with pytest.raises(RuntimeError, match="not a model score"):
        parse_grade("", 1)
    with pytest.raises(ValidationError):
        parse_grade("{}", 0)


@pytest.mark.parametrize(
    "module,mode,env",
    [
        ("module_4_jetbrains_mcp", "bundled", "IDE_INSPECTIONS_IMAGE"),
    ],
)
def test_modules_load_real_gym_models_and_image_definitions(
    module: str, mode: str, env: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(env, "registry/image:tag")
    path = ROOT / "experiments" / module
    config = OmegaConf.load(path / "config.yaml")
    agent = OmegaConf.to_container(config.ide_agent.responses_api_agents.inspection_agent, resolve=True)
    agent.update(host="127.0.0.1", port=9000, name="ide_agent")
    parsed = IdeAgentConfig.model_validate(agent)
    assert parsed.ide.mode == mode
    assert parsed.ide.image == PYCHARM_IMAGE
    request = InspectionRunRequest.model_validate_json((path / "data/task.jsonl").read_text())
    assert request.task_id == "ide-inspections"
    image = yaml.safe_load((path / "image.yaml").read_text())["images"][0]
    assert image["context_uri"].startswith("gs://")
    assert image["platforms"] == ["linux/amd64"]
    assert 'PYCHARM_VERSION="2026.1.1"' in image["base_dockerfile"]
    assert "USER appuser" in image["base_dockerfile"]


@pytest.mark.parametrize("failure", ["none", "initialization", "evaluation", "deletion"])
async def test_sandbox_deleted_after_success_or_failure(failure: str, monkeypatch: pytest.MonkeyPatch) -> None:
    from idegym.api.orchestrator.servers import ErrorResponse

    from experiments.ide_mcp import transport

    for key, value in {
        "IDEGYM_ORCHESTRATOR_URL": "https://example.test",
        "IDEGYM_AUTH_USERNAME": "test-user",
        "IDEGYM_AUTH_PASSWORD": "test-password",
    }.items():
        monkeypatch.setenv(key, value)
    server = SimpleNamespace(server_id=42)
    stopped = ErrorResponse(status_code=500) if failure == "deletion" else None
    client = SimpleNamespace(
        start_server=AsyncMock(return_value=server),
        stop_server=AsyncMock(return_value=stopped),
        client_id="test-client",
    )
    session = SimpleNamespace(
        initialize=AsyncMock(side_effect=RuntimeError("init failed") if failure == "initialization" else None)
    )

    @asynccontextmanager
    async def fake_client(**kwargs):
        yield client

    @asynccontextmanager
    async def fake_http(**kwargs):
        yield None

    @asynccontextmanager
    async def fake_stream(*args, **kwargs):
        # Behind the deployed ingress, /mcp redirects to insecure http://.../mcp/.
        assert args[0] == "https://example.test/mcp/"
        yield None, None, None

    @asynccontextmanager
    async def fake_session(*args):
        yield session

    monkeypatch.setattr(transport, "IdeGYMClient", fake_client)
    monkeypatch.setattr(transport.httpx, "AsyncClient", fake_http)
    monkeypatch.setattr(transport, "streamable_http_client", fake_stream)
    monkeypatch.setattr(transport, "ClientSession", fake_session)

    async def run():
        async with transport.connect_ide(IdeSettings(mode="bundled", image="registry/image:tag")) as connection:
            assert connection.scope == {"client_id": "test-client", "server_id": 42}
            if failure == "evaluation":
                raise RuntimeError("evaluation failed")

    if failure == "none":
        await run()
    else:
        with pytest.raises(RuntimeError):
            await run()
    client.stop_server.assert_awaited_once()
    assert client.start_server.call_args.kwargs["reuse_strategy"].value == "NONE"
    assert client.start_server.call_args.kwargs["run_as_root"] is False


async def test_inspection_check_rejects_a_missing_baseline(monkeypatch, tmp_path) -> None:
    connection = SimpleNamespace(
        discover=AsyncMock(return_value=[ToolInfo(name="pycharm_get_file_problems", input_schema={})]),
    )

    @asynccontextmanager
    async def connect(settings):
        yield connection

    monkeypatch.setattr("experiments.ide_mcp.cli.connect_ide", connect)
    monkeypatch.setattr("experiments.ide_mcp.cli.monotonic", Mock(side_effect=[0, 2]))
    monkeypatch.setattr("experiments.module_4_jetbrains_mcp.task.seed", AsyncMock())
    monkeypatch.setattr(
        "experiments.module_4_jetbrains_mcp.task.inspect",
        AsyncMock(return_value={"pricing.py": [], "receipt.py": []}),
    )
    with pytest.raises(RuntimeError, match="Baseline invalid"):
        await check_image(IdeSettings(mode="bundled", image="registry/image:tag", ready_timeout=1), tmp_path)


@pytest.mark.parametrize("request_type", [BashRunRequest, TerminalRunRequest, InspectionRunRequest])
def test_run_validation_keeps_model_capture_correlation(request_type: type[BaseRunRequest]) -> None:
    body = request_type.model_validate(
        {
            "responses_create_params": {"input": "Repair the project"},
            TASK_INDEX_KEY_NAME: 3,
            ROLLOUT_INDEX_KEY_NAME: 2,
            ATTEMPT_INDEX_KEY_NAME: 1,
        }
    )
    agent = TerminalAgent.model_construct(
        server_client=SimpleNamespace(global_config_dict={"observability_enabled": True})
    )
    assert agent.url_path_for_run("/v1/responses", body) == "/ng-rollout/3-2-a1/v1/responses"


async def test_empty_reply_after_tool_read_is_incomplete(tmp_path: Path) -> None:
    # Matches the malformed OpenRouter reply: a tool marker in reasoning, no actual call.
    provider_reply = NeMoGymChatCompletion.model_validate(
        {
            "id": "provider-empty",
            "created": 0,
            "model": "test",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": None, "reasoning": "<tool_call|>"},
                }
            ],
        }
    )
    empty_reply = ResponsesConverter(
        return_token_id_information=False, uses_reasoning_parser=False
    ).chat_completion_to_response(
        NeMoGymResponseCreateParamsNonStreaming(input="Repair the project", model="test"), provider_reply
    )
    model = AsyncMock(
        side_effect=[
            response(
                [
                    {
                        "type": "function_call",
                        "name": "bash",
                        "arguments": '{"command":"cat calculator.py"}',
                        "call_id": "read-1",
                        "id": "read-1",
                    }
                ]
            ),
            empty_reply,
        ]
    )
    connection = SimpleNamespace(
        call=AsyncMock(return_value=ToolResult(content=[{"type": "text", "text": "def add(a, b): return a + b"}]))
    )
    result, attempts, successes = await run_episode(
        NeMoGymResponseCreateParamsNonStreaming(input="Rename add to sum_numbers"),
        [ToolInfo(name="bash", input_schema={})],
        connection,
        model,
        20,
        tmp_path / "trajectory.jsonl",
    )
    assert (attempts, successes) == (1, 1)
    assert model.await_count == 2
    assert result.status == "incomplete"
