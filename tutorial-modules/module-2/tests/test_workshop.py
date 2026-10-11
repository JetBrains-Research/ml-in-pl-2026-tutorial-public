import json
from pathlib import Path

import pytest
from click.testing import CliRunner
from nemo_gym.config_types import DatasetConfig
from omegaconf import OmegaConf

from experiments.module_1_python import workshop
from experiments.module_1_python.cli import cli
from experiments.module_1_python.workshop_support import PythonVerifyRequest, PythonVerifyResponse


@pytest.fixture
def workshop_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setattr(workshop, "configure", lambda module: None)
    monkeypatch.setattr(workshop, "prepare_runtime", lambda components: None)
    for name, value in {
        "POC_MODEL_NAME": "vertex_ai/google/gemma-4-26b-a4b-it-maas",
        "POC_MODEL_BASE_URL": "https://model.example/v1",
        "POC_MODEL_API_KEY": "private-model-key",
        "POC_RUNTIME_DIR": str(tmp_path / ".runtime"),
        "IDEGYM_ORCHESTRATOR_URL": "https://idegym.example",
        "IDEGYM_AUTH_USERNAME": "private-user",
        "IDEGYM_AUTH_PASSWORD": "private-password",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("MODEL_PROVIDER", raising=False)
    # Restore output-specific settings changed by the CLI after each test.
    monkeypatch.setenv("POC_LOG_DIR", "")
    monkeypatch.setenv("POC_WORKSHOP_EVENTS", "")
    return tmp_path / "run"


def fake_gym(command: list[str], log, events: Path, *, timeout_s: float | None = None) -> int:
    """Stand in for hosted generation; keep task/config/output handling real."""
    config = OmegaConf.load(command[command.index("--config") + 1])
    dataset = config.idegym_python_agent.responses_api_agents.simple_agent.datasets[0]
    parsed = DatasetConfig.model_validate(OmegaConf.to_container(dataset, resolve=True))
    tasks = [json.loads(line) for line in Path(parsed.jsonl_fpath).read_text().splitlines()]
    rows = []
    for task in tasks * int(command[command.index("--num-repeats") + 1]):
        body = PythonVerifyRequest.model_validate(
            task
            | {
                "response": {
                    "id": "test-response",
                    "created_at": 0,
                    "object": "response",
                    "model": "test",
                    "parallel_tool_calls": False,
                    "tool_choice": "auto",
                    "tools": [],
                    "output": [
                        {
                            "type": "message",
                            "id": "test-message",
                            "role": "assistant",
                            "status": "completed",
                            "content": [
                                {"type": "output_text", "text": "def solve(data): return 1", "annotations": []}
                            ],
                        }
                    ],
                }
            }
        )
        cases = body.verifier_metadata.test_cases
        result = PythonVerifyResponse(
            **body.model_dump(),
            reward=1,
            passed_tests=len(cases),
            total_tests=len(cases),
            test_results=[
                {
                    "index": index,
                    "passed": True,
                    "stdout": json.dumps(case.expected),
                    "stderr": "",
                    "error_type": "",
                    "return_code": 0,
                }
                for index, case in enumerate(cases)
            ],
        )
        rows.append(result.model_dump_json())
    Path(command[command.index("--output") + 1]).write_text("\n".join(rows) + "\n")
    return 0


@pytest.mark.parametrize("count,repeats", [(1, 1), (3, 1), (5, 1), (2, 2)])
def test_workshop_task_selection_and_results(
    workshop_environment: Path, monkeypatch: pytest.MonkeyPatch, count: int, repeats: int
) -> None:
    monkeypatch.setattr(workshop, "run_with_progress", fake_gym)
    args = ["--output-dir", str(workshop_environment), "--limit", str(count), "--repeats", str(repeats)]
    result = CliRunner().invoke(workshop.main, args)
    assert result.exit_code == 0, result.output
    summary = json.loads((workshop_environment / "summary.json").read_text())
    assert summary["rollouts"] == count * repeats
    assert summary["samples"] == count
    assert summary["repeats"] == repeats
    assert summary["provider"] == "litellm"
    assert summary["mean_reward"] == 1
    assert "Generated solution:" in result.output
    assert "[3/3] Evaluation complete" in result.output
    assert result.output.count("PASS  solve(") == count * repeats * 3
    tasks = [json.loads(line) for line in (workshop_environment / "tasks.jsonl").read_text().splitlines()]
    if count == 1:
        assert tasks[0]["verifier_metadata"]["task_id"] == "factorial"
    saved_config = (workshop_environment / "config.yaml").read_text()
    assert "${oc.env:POC_MODEL_API_KEY}" in saved_config
    for secret in ("private-model-key", "private-password", "private-user"):
        assert secret not in saved_config + result.output


def test_edits_to_sample_file_and_timeout_reach_gym(
    workshop_environment: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = json.loads(workshop.SAMPLES_FILE.read_text().splitlines()[0])
    task["responses_create_params"]["input"][1]["content"] = "Use recursion to compute factorial."
    task["verifier_metadata"]["test_cases"].append({"input": 1, "expected": 1})
    samples_file = workshop_environment.parent / "custom.jsonl"
    samples_file.write_text(json.dumps(task) + "\n")
    monkeypatch.setattr(workshop, "run_with_progress", fake_gym)
    result = CliRunner().invoke(
        workshop.main,
        ["--output-dir", str(workshop_environment), "--test-timeout", "1", "--samples-file", str(samples_file)],
    )
    assert result.exit_code == 0, result.output
    saved = json.loads((workshop_environment / "tasks.jsonl").read_text())
    assert saved["verifier_metadata"] == task["verifier_metadata"]
    assert (
        saved["responses_create_params"]["input"][1]["content"]
        == task["responses_create_params"]["input"][1]["content"]
    )
    config = OmegaConf.load(workshop_environment / "config.yaml")
    assert config.idegym_python.resources_servers.idegym_python.test_timeout_s == 1
    assert "4/4 cases passed" in result.output
    assert "Use recursion" in result.output


def test_failed_gym_run_has_no_score(workshop_environment: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(workshop, "run_with_progress", lambda *args, **kwargs: 1)
    result = CliRunner().invoke(workshop.main, ["--output-dir", str(workshop_environment)])
    assert result.exit_code == 1
    assert "Evaluation could not finish" in result.output
    assert "this is not reward 0" in result.output
    assert "Mean reward:" not in result.output
    assert not (workshop_environment / "summary.json").exists()


def test_gym_deadline_is_an_incomplete_evaluation(workshop_environment: Path, monkeypatch) -> None:
    import subprocess

    def stalled(command, log, events, *, timeout_s):
        assert timeout_s == 42
        raise subprocess.TimeoutExpired(command, timeout_s)

    monkeypatch.setattr(workshop, "run_with_progress", stalled)
    result = CliRunner().invoke(workshop.main, ["--run-timeout", "42", "--output-dir", str(workshop_environment)])
    assert result.exit_code == 1
    assert "exceeded the 42s run deadline" in result.output
    assert "not reward 0" in result.output
    assert "Artifacts and logs:" in result.output
    assert not (workshop_environment / "summary.json").exists()


def test_progress_heartbeat_while_gym_has_no_events(tmp_path, monkeypatch, capsys) -> None:
    from experiments.module_1_python import workshop_support

    events = tmp_path / "progress.jsonl"
    events.touch()

    def waiting(command, log, *, timeout_s):
        import time

        time.sleep(0.5)
        return 0

    monkeypatch.setattr(workshop_support, "run_logged", waiting)
    with (tmp_path / "gym.log").open("w") as log:
        assert workshop_support.run_with_progress([], log, events, heartbeat_s=0.1) == 0
    assert "Still running: Waiting for local Ray startup" in capsys.readouterr().out


def test_wrong_answer_is_a_completed_zero_reward_run(
    workshop_environment: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def wrong_answer(command, log, events, **kwargs):
        fake_gym(command, log, events)
        path = Path(command[command.index("--output") + 1])
        row = json.loads(path.read_text())
        row.update(reward=0, passed_tests=2, failure_reason="test_failure")
        row["test_results"][0].update(passed=False, stdout="0")
        path.write_text(json.dumps(row) + "\n")
        return 0

    monkeypatch.setattr(workshop, "run_with_progress", wrong_answer)
    result = CliRunner().invoke(workshop.main, ["--output-dir", str(workshop_environment)])
    assert result.exit_code == 0, result.output
    assert "FAIL  solve(0) -> 0; expected 1" in result.output
    assert "Reward: 0.0 — 2/3 cases passed" in result.output
    assert json.loads((workshop_environment / "summary.json").read_text())["mean_reward"] == 0


def test_existing_output_is_preserved(workshop_environment: Path) -> None:
    workshop_environment.mkdir()
    marker = workshop_environment / "summary.json"
    marker.write_text("existing evidence")
    result = CliRunner().invoke(workshop.main, ["--output-dir", str(workshop_environment)])
    assert result.exit_code == 1
    assert "already exists" in result.output
    assert marker.read_text() == "existing evidence"


@pytest.mark.parametrize(
    "contents,limit,message",
    [
        ("\n\n", 1, "No samples found"),
        ("{broken json}\n", 1, "at line 1"),
        ('{"responses_create_params": {"input": "hello"}}\n', 1, "at line 1"),
        (workshop.SAMPLES_FILE.read_text(), 6, "contains only 5"),
    ],
)
def test_invalid_samples_fail_before_starting_gym(
    workshop_environment: Path, monkeypatch: pytest.MonkeyPatch, contents: str, limit: int, message: str
) -> None:
    from unittest.mock import Mock

    start = Mock()
    monkeypatch.setattr(workshop, "run_with_progress", start)
    path = workshop_environment.parent / "samples.jsonl"
    path.write_text(contents)
    result = CliRunner().invoke(
        workshop.main, ["--samples-file", str(path), "--limit", str(limit), "--output-dir", str(workshop_environment)]
    )
    assert result.exit_code == 1
    assert message in result.output
    assert json.loads((workshop_environment / "run.json").read_text())["status"] == "failed"
    start.assert_not_called()


def test_blank_lines_do_not_count_as_samples(tmp_path: Path) -> None:
    lines = workshop.SAMPLES_FILE.read_text().splitlines()
    path = tmp_path / "samples.jsonl"
    path.write_text("\n" + lines[0] + "\n\n" + lines[1] + "\n")
    assert [sample.verifier_metadata.task_id for sample in workshop.load_samples(path, 2)] == ["factorial", "sum-list"]


def test_module_has_one_run_command() -> None:
    assert set(cli.commands) == {"run"}
    result = CliRunner().invoke(cli, ["run", "--help"])
    assert result.exit_code == 0
    assert "--limit" in result.output and "--samples-file" in result.output
    assert "--all-tasks" not in result.output
