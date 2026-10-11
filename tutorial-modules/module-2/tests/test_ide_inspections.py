import json
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from experiments.ide_mcp.transport import IdeSettings, ToolResult
from experiments.module_4_jetbrains_mcp.evaluation import (
    PATH_FIELDS,
    InspectionChecks,
    ScopedInspectionConnection,
    inspected_before_and_after_edits,
)
from experiments.module_4_jetbrains_mcp.task import (
    BROKEN,
    FILES,
    FIXED,
    VERIFY_SCRIPT,
    inspect,
    parse_problems,
)


def diagnostic(payload):
    return ToolResult(content=[{"type": "text", "text": json.dumps(payload)}])


def test_observed_inspection_response_and_empty_results():
    problems = [{"severity": "WEAK WARNING", "description": "Unused parameter", "line": 1, "column": 51}]
    assert parse_problems(diagnostic({"filePath": "pricing.py", "errors": problems})) == problems
    assert parse_problems(diagnostic({"filePath": "pricing.py", "errors": []})) == []
    for payload in ({}, {"errors": "timeout"}, {"errors": ["invalid"]}):
        with pytest.raises(RuntimeError):
            parse_problems(diagnostic(payload))
    with pytest.raises(RuntimeError, match="inspection call failed"):
        parse_problems(ToolResult(is_error=True, content=[]))


async def test_harness_calls_each_file_and_preserves_raw_error(tmp_path):
    connection = SimpleNamespace(
        call=AsyncMock(
            side_effect=[
                diagnostic({"errors": [{"severity": "ERROR"}]}),
                ToolResult(is_error=True, content=[]),
            ]
        )
    )
    with pytest.raises(RuntimeError):
        await inspect(connection, tmp_path, "before")
    assert connection.call.call_count == 2
    assert (tmp_path / "before-receipt.py.json").exists()
    for call in connection.call.call_args_list:
        assert call.args[0] == "pycharm_get_file_problems"
        assert call.args[1]["errorsOnly"] is False


def entry(tool, path):
    return dict(tool=tool, arguments=json.dumps({"filePath": path}), is_error=False)


def test_model_must_inspect_both_files_before_and_after_editing():
    before = [entry("pycharm_get_file_problems", name) for name in FILES]
    edit = entry("pycharm_replace_text_in_file", "pricing.py")
    assert inspected_before_and_after_edits([*before, edit, *before])
    assert not inspected_before_and_after_edits([edit, *before])
    assert not inspected_before_and_after_edits([*before, edit])
    assert not inspected_before_and_after_edits([*before, edit, before[0]])
    assert not inspected_before_and_after_edits(before)


async def test_model_cannot_modify_ide_settings_or_use_terminal():
    connection = ScopedInspectionConnection(
        SimpleNamespace(),
        SimpleNamespace(server_id=1),
        "test",
        IdeSettings(mode="bundled", image="registry/image:tag"),
    )
    for name, path in (
        ("pycharm_replace_text_in_file", ".idea/inspectionProfiles/profile.xml"),
        ("pycharm_execute_terminal_command", "pricing.py"),
    ):
        assert (await connection.call(name, {PATH_FIELDS.get(name, "filePath"): path})).is_error


def test_all_checks_required_for_reward():
    fields = dict(baseline_detected=True, inspections_clean=True, behavior_preserved=True, model_used_inspections=True)
    assert InspectionChecks(**fields).reward == 1
    for name in fields:
        assert InspectionChecks(**(fields | {name: False})).reward == 0


def test_reference_behavior_and_broken_fixture(tmp_path):
    script = VERIFY_SCRIPT.replace("/home/appuser/work", str(tmp_path))
    for files, expected in ((BROKEN, False), (FIXED, True)):
        for name, source in files.items():
            (tmp_path / name).write_text(source)
        result = subprocess.run([sys.executable, "-I", "-c", script], capture_output=True, text=True, check=True)
        checks = json.loads(result.stdout)
        assert (checks["pricing"] and checks["receipt"]) is expected
    (tmp_path / "pricing.py").write_text(FIXED["pricing.py"] + "# noinspection PyUnusedLocal\n")
    result = subprocess.run([sys.executable, "-I", "-c", script], capture_output=True, text=True, check=True)
    assert not json.loads(result.stdout)["no_suppressions"]


async def test_scoped_tools_forward_each_observed_path_field():
    from unittest.mock import AsyncMock

    from mcp.types import CallToolResult

    session = SimpleNamespace(
        call_tool=AsyncMock(
            return_value=CallToolResult(
                content=[],
                structuredContent={"content": [], "is_error": False},
            )
        )
    )
    connection = ScopedInspectionConnection(
        session,
        SimpleNamespace(server_id=1),
        "test",
        IdeSettings(mode="bundled", image="registry/image:tag"),
    )
    for name, field in PATH_FIELDS.items():
        result = await connection.call(name, {field: "pricing.py"})
        assert not result.is_error
    assert session.call_tool.call_count == 4


@pytest.mark.parametrize("remaining", [[], [{"severity": "WARNING", "description": "Still broken"}]])
@pytest.mark.parametrize("golden", [True, False])
async def test_fresh_grader_controls_reward_and_never_runs_verifier_in_agent(monkeypatch, tmp_path, remaining, golden):
    result, events, connections = await run_mock_evaluation(monkeypatch, tmp_path, remaining=remaining, golden=golden)
    assert result.reward == (0 if remaining else 1)
    assert events == ["open-agent", "open-grader", "close-agent", "close-grader"]
    agent, grader = connections
    from experiments.ide_mcp.task import python_command

    verifier = python_command(VERIFY_SCRIPT)
    assert not any(call.kwargs["script"] == verifier for call in agent.server.execute_bash.call_args_list)
    assert any(call.kwargs["script"] == verifier for call in grader.server.execute_bash.call_args_list)
    assert [call.args[1]["filePath"] for call in agent.call.call_args_list] == list(FILES)
    assert [call.args[1]["filePath"] for call in grader.call.call_args_list] == [*FILES, *FILES]
    assert json.loads((tmp_path / "final-files.json").read_text()) == FIXED
    identities = json.loads((tmp_path / "sandboxes.json").read_text())
    assert identities["agent"]["server_id"] != identities["grader"]["server_id"]
    for name in [
        "inspections-before.json",
        "inspections-grader-before.json",
        "inspections-after.json",
        "verification.json",
    ]:
        assert (tmp_path / name).exists()


async def test_missing_submission_file_is_a_failed_grade(monkeypatch, tmp_path):
    result, _, connections = await run_mock_evaluation(monkeypatch, tmp_path, submission=FIXED | {"receipt.py": None})
    assert result.reward == 0
    assert not result.behavior["files_present"]
    assert result.after_counts["receipt.py"] == 1
    assert len(connections[1].call.call_args_list) == 3


@pytest.mark.parametrize("failure", ["transfer", "verifier"])
async def test_grading_failure_cleans_up_both_sandboxes(monkeypatch, tmp_path, failure):
    events = []
    with pytest.raises(RuntimeError, match="transfer submission|verifier did not execute"):
        await run_mock_evaluation(monkeypatch, tmp_path, failure=failure, events=events)
    assert events == ["open-agent", "open-grader", "close-agent", "close-grader"]
    assert not (tmp_path / "verification.json").exists()


async def run_mock_evaluation(
    monkeypatch, tmp_path, *, remaining=None, golden=True, submission=None, failure=None, events=None
):
    from contextlib import asynccontextmanager

    from experiments.ide_mcp.task import python_command
    from experiments.ide_mcp.transport import ToolInfo
    from experiments.module_4_jetbrains_mcp import workshop as evaluation

    remaining = [] if remaining is None else remaining
    submission = FIXED if submission is None else submission
    events = [] if events is None else events
    connections = []
    for role in ("agent", "grader"):

        async def execute(*, script, role=role, **kwargs):
            if "Submission contains a symlink" in script:
                return SimpleNamespace(exit_code=0, stdout=json.dumps(submission))
            if script == python_command(VERIFY_SCRIPT):
                return SimpleNamespace(
                    exit_code=int(failure == "verifier"),
                    stdout=json.dumps(
                        dict(
                            files_present=all(value is not None for value in submission.values()),
                            no_suppressions=True,
                            pricing=True,
                            receipt=True,
                        )
                    ),
                )
            if role == "grader" and script == python_command(
                evaluation.write_files_script(
                    {name: source for name, source in submission.items() if source is not None}, clean=True
                )
            ):
                return SimpleNamespace(exit_code=int(failure == "transfer"))
            return SimpleNamespace(exit_code=0)

        calls = [diagnostic({"errors": [{"severity": "WARNING"}]}), diagnostic({"errors": [{"severity": "ERROR"}]})]
        if role == "grader":
            calls += [diagnostic({"errors": remaining}) for name in FILES if submission[name] is not None]
        connections.append(
            SimpleNamespace(
                discover=AsyncMock(return_value=[ToolInfo(name="pycharm_get_file_problems", input_schema={})]),
                call=AsyncMock(side_effect=calls),
                session=SimpleNamespace(),
                scope={"client_id": role, "server_id": len(connections) + 1},
                server=SimpleNamespace(server_id=len(connections) + 1, execute_bash=AsyncMock(side_effect=execute)),
            )
        )

    @asynccontextmanager
    async def connect(settings):
        role = "agent" if not events else "grader"
        connection = connections[0 if role == "agent" else 1]
        events.append("open-" + role)
        try:
            yield connection
        finally:
            events.append("close-" + role)

    async def episode(*args):
        before = [entry("pycharm_get_file_problems", name) for name in FILES]
        trace = [*before, entry("pycharm_replace_text_in_file", "pricing.py"), *before]
        (tmp_path / "trajectory.jsonl").write_text("\n".join(json.dumps(row) for row in trace))
        return SimpleNamespace(), 5, 5

    monkeypatch.setattr(evaluation, "connect_ide", connect)
    monkeypatch.setattr(evaluation, "run_episode", episode)
    result, _ = await evaluation.evaluate(
        IdeSettings(mode="bundled", image="registry/image:tag"),
        tmp_path,
        params=None if golden else SimpleNamespace(),
        model_call=None if golden else AsyncMock(),
    )
    return result, events, connections
