"""Run bundles retain native Gym evidence, console output, and failed-run diagnostics."""

import json
import os
import re
from contextlib import nullcontext
from pathlib import Path

import click
import pytest
from click.testing import CliRunner
from fastapi import FastAPI
from fastapi.testclient import TestClient
from nemo_gym.base_responses_api_model import ModelCallCaptureConfig, install_model_call_capture
from omegaconf import OmegaConf
from ray._common.utils import get_default_ray_temp_dir

from experiments import artifacts
from experiments.common import progress
from experiments.workshop_support import event, run_reference

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "module", ["module_1_python", "module_2_direct_bash", "module_3_openenv_bash", "module_4_jetbrains_mcp"]
)
def test_run_bundle_captures_gym_model_exchange_and_console(module, tmp_path):
    module_root = ROOT / "experiments" / module
    output = tmp_path / "run"

    @click.command()
    @click.option("--output-dir", type=click.Path(path_type=Path))
    @artifacts.record_run(module_root)
    def command(output_dir):
        configuration = OmegaConf.load(output_dir / "config.yaml")
        capture = ModelCallCaptureConfig(
            observability_enabled=configuration.observability_enabled,
            model_call_capture_dir=configuration.model_call_capture_dir,
        )
        assert capture.model_call_capture_dir.parent == output_dir
        assert Path(configuration.nemo_gym_log_dir).parent == output_dir
        assert Path(configuration.results_dir) == output_dir
        assert not Path(os.environ["POC_RUNTIME_DIR"]).is_relative_to(output_dir)
        assert Path(configuration.uv_venv_dir) == Path(os.environ["POC_RUNTIME_DIR"])
        app = FastAPI()

        @app.post("/v1/responses")
        def response(body: dict):
            assert body["input"] == "hello"
            return {"model": "local-test", "output": [], "usage": {"input_tokens": 2, "output_tokens": 1}}

        install_model_call_capture(app, capture, model_server_name="policy_model")
        with TestClient(app) as client:
            reply = client.post("/ng-rollout/local-0/v1/responses", json={"model": "local-test", "input": "hello"})
        assert reply.status_code == 200
        click.echo("Model call completed")
        click.echo("Diagnostic message", err=True)
        progress("local-0", "Ready for grading")
        (output_dir / "summary.json").write_text(json.dumps({"reward": 1}))

    before = os.environ.get("POC_RUN_DIR")
    result = CliRunner().invoke(command, ["--output-dir", str(output)])
    assert result.exit_code == 0, result.exception
    assert os.environ.get("POC_RUN_DIR") == before
    console = (output / "console.log").read_text()
    assert "Model call completed" in console and "Diagnostic message" in console
    assert "Model call completed" in result.output
    assert json.loads((output / "run.json").read_text())["status"] == "completed"
    captures = list((output / "model-calls").glob("*.capture.jsonl"))
    assert len(captures) == 1
    assert '"hello"' in captures[0].read_text()
    assert json.loads((output / "progress.jsonl").read_text())["timestamp"]
    assert "model-calls/local-0.capture.jsonl" in (output / "README.md").read_text()
    assert not (output / "source.zip").exists()
    assert not (output / "uv.lock").exists()
    assert not (output / "pyproject.snapshot.toml").exists()
    assert not (output / "pyproject.toml").exists()


@pytest.mark.parametrize(
    "error,status", [(RuntimeError("service failed"), "failed"), (KeyboardInterrupt(), "interrupted")]
)
def test_failed_run_keeps_partial_evidence_and_restores_environment(error, status, tmp_path):
    output = tmp_path / "run"

    @click.command()
    @click.option("--output-dir", type=click.Path(path_type=Path))
    @artifacts.record_run(ROOT / "experiments/module_1_python")
    def command(output_dir):
        click.echo("Preparing environment")
        (output_dir / "partial.json").write_text('{"prepared": true}')
        raise error

    before = os.environ.get("POC_RUN_DIR")
    result = CliRunner().invoke(command, ["--output-dir", str(output)])
    assert result.exit_code != 0
    assert os.environ.get("POC_RUN_DIR") == before
    assert json.loads((output / "run.json").read_text())["status"] == status
    assert (output / "error.log").exists()
    assert "Preparing environment" in (output / "console.log").read_text()
    assert "partial.json" in (output / "README.md").read_text()
    assert not (output / "summary.json").exists()


def test_default_output_groups_module_and_utc_timestamp(tmp_path, monkeypatch):
    monkeypatch.setattr(artifacts, "ROOT", tmp_path)
    module = ROOT / "experiments/module_2_direct_bash"
    output = artifacts.new_output(module, None, "golden")
    assert output.parent == tmp_path / "results/module_2_direct_bash"
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}\.\d{6}Z-golden", output.name)
    assert output.stat().st_mode & 0o777 == 0o700


def test_fast_reference_run_relays_final_progress_and_keeps_sdk_logs_private(tmp_path, capsys):
    async def episode():
        print("Raw SDK diagnostic")
        event("Reference grading complete")
        return 1

    assert run_reference(episode(), 5, tmp_path) == 1
    console = capsys.readouterr().out
    assert "Reference grading complete" in console
    assert "Raw SDK diagnostic" not in console
    assert "Raw SDK diagnostic" in (tmp_path / "reference.log").read_text()


def test_artifact_index_failure_still_restores_run_context(tmp_path, monkeypatch):
    output = tmp_path / "run"

    def fail_index(output, record):
        raise OSError("disk full")

    monkeypatch.setattr(artifacts, "write_index", fail_index)

    @click.command()
    @click.option("--output-dir", type=click.Path(path_type=Path))
    @artifacts.record_run(ROOT / "experiments/module_1_python")
    def command(output_dir):
        click.echo("Completed evaluation")

    before = os.environ.get("POC_RUN_DIR")
    result = CliRunner().invoke(command, ["--output-dir", str(output)])
    assert isinstance(result.exception, OSError)
    assert os.environ.get("POC_RUN_DIR") == before
    assert artifacts.current_output() is None


def test_input_snapshot_preserves_environment_references_without_credentials(tmp_path, monkeypatch):
    for name in ("POC_RUNTIME_DIR", "POC_RUN_DIR", "POC_LOG_DIR"):
        monkeypatch.delenv(name, raising=False)
    artifacts.snapshot_inputs(ROOT / "experiments/module_1_python", tmp_path)
    config = (tmp_path / "config.yaml").read_text()
    assert "${oc.env:IDEGYM_AUTH_PASSWORD}" in config
    assert "${oc.env:POC_MODEL_API_KEY}" in config


@pytest.mark.parametrize("fails", [False, True])
def test_gym_runtime_is_temporary_on_success_and_failure(fails, tmp_path, monkeypatch):
    monkeypatch.setenv("RAY_TMPDIR", "/tmp/previous-ray-root")
    monkeypatch.setenv("POC_RUNTIME_DIR", "/tmp/previous-components")
    with pytest.raises(RuntimeError) if fails else nullcontext():
        with artifacts.run_runtime():
            alias = Path(os.environ["RAY_TMPDIR"])
            ray_root = Path(get_default_ray_temp_dir())
            assert ray_root == alias / "ray"
            assert not ray_root.is_relative_to(tmp_path)
            component = Path(os.environ["POC_RUNTIME_DIR"]) / "test"
            component.mkdir(parents=True)
            (component / ".venv").symlink_to("/missing/container/venv", target_is_directory=True)
            (ray_root / "logs").mkdir(parents=True)
            (ray_root / "logs/gcs_server.out").write_text("Native Ray diagnostic")
            if fails:
                raise RuntimeError("Ray startup failed")
    assert not alias.exists()
    assert os.environ["RAY_TMPDIR"] == "/tmp/previous-ray-root"
    assert os.environ["POC_RUNTIME_DIR"] == "/tmp/previous-components"


def test_artifact_index_skips_ray_sockets_on_docker_bind_mount(tmp_path, monkeypatch):
    session = tmp_path / "ray/session"
    socket_path = session / "sockets/plasma_store"
    socket_path.parent.mkdir(parents=True)
    socket_path.touch()
    log = session / "logs/raylet.out"
    log.parent.mkdir()
    log.write_text("Native Ray log")
    original = Path.is_file

    def is_file(path):
        if path == socket_path:
            raise OSError(95, "Operation not supported")
        return original(path)

    monkeypatch.setattr(Path, "is_file", is_file)
    artifacts.write_index(
        tmp_path, {"module": "module_1_python", "mode": "model", "status": "completed", "started_at": "now"}
    )
    index = (tmp_path / "README.md").read_text()
    assert "ray/session/logs/raylet.out" in index
    assert "plasma_store" not in index


@pytest.mark.parametrize("fails", [False, True])
@pytest.mark.parametrize("golden", [False, True])
def test_run_keeps_runtime_outside_bundle_and_cleans_it(fails, golden, tmp_path):
    output = tmp_path / "run"
    runtime_paths = []

    @click.command()
    @click.option("--output-dir", type=click.Path(path_type=Path))
    @click.option("--golden", is_flag=True)
    @artifacts.record_run(ROOT / "experiments/module_1_python")
    def command(output_dir, golden):
        runtime = Path(os.environ["POC_RUNTIME_DIR"])
        runtime_paths.append(runtime)
        component = runtime / "responses_api_agents/simple_agent"
        component.mkdir(parents=True)
        (component / ".venv").symlink_to("/missing/container/venv", target_is_directory=True)
        if fails:
            raise RuntimeError("evaluation failed")

    args = ["--output-dir", str(output), *(["--golden"] if golden else [])]
    result = CliRunner().invoke(command, args)
    assert (result.exit_code != 0) == fails
    assert len(runtime_paths) == 1
    assert not runtime_paths[0].exists()
    assert not (output / "runtime").exists()
    assert not (output / "ray").exists()
    assert not list(output.rglob(".venv"))
