"""Cloud build submission, resume, failure reporting, and client cleanup."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import click
import pytest

from experiments import image_build


@pytest.fixture
def build_environment(monkeypatch, tmp_path):
    monkeypatch.setattr(image_build, "ROOT", tmp_path)
    monkeypatch.setenv("IDEGYM_ORCHESTRATOR_URL", "https://workshop.example")
    monkeypatch.setenv("IDEGYM_AUTH_USERNAME", "workshop-user")
    monkeypatch.setenv("IDEGYM_AUTH_PASSWORD", "private-password")
    monkeypatch.setenv("IDEGYM_NAMESPACE", "idegym")
    module = tmp_path / "module_4_jetbrains_mcp"
    module.mkdir()
    (module / "image.yaml").write_text("images: [{name: generic-ide, base: debian:bookworm}]\n")
    client = SimpleNamespace(close=AsyncMock())
    monkeypatch.setattr(image_build, "get_global_aiohttp_client", lambda *_: client)
    monkeypatch.setattr(image_build, "get_response_json", AsyncMock(side_effect=lambda response: response.payload))
    monkeypatch.setattr(image_build.asyncio, "sleep", AsyncMock())
    return module, client


def response(payload, status=200):
    return SimpleNamespace(status=status, payload=payload, text=AsyncMock(return_value=str(payload)))


@pytest.mark.parametrize("resume", [False, True])
async def test_cloud_build_records_result_and_resume_does_not_submit(build_environment, monkeypatch, resume):
    module, client = build_environment
    replies = [] if resume else [response({"job_names": ["cloud-job"]})]
    replies += [
        response({"job_name": "cloud-job", "status": "in_progress"}),
        response({"job_name": "cloud-job", "status": "success", "tag": "google.example/ide:hash"}),
    ]
    request = AsyncMock(side_effect=replies)
    monkeypatch.setattr(image_build, "request", request)
    await image_build.build_image(
        module_root=module, job_name="cloud-job" if resume else None, metadata={"runtime": "pycharm-mcp"}
    )
    calls = request.call_args_list
    if not resume:
        assert calls[0].args == ("POST", "https://workshop.example/api/build-push-images")
        payload = calls[0].kwargs["json"]
        assert payload == {"namespace": "idegym", "yaml_content": (module / "image.yaml").read_text()}
        assert "private-password" not in json.dumps(payload)
        calls = calls[1:]
    assert all(call.args == ("GET", "https://workshop.example/api/jobs/status/cloud-job") for call in calls)
    record = json.loads((module.parent / ".runtime" / module.name / "image.json").read_text())
    assert record["status"] == "SUCCESS" and record["image"] == "google.example/ide:hash"
    assert record["runtime"] == "pycharm-mcp"
    client.close.assert_awaited_once()


@pytest.mark.parametrize("submission_failure", [False, True])
async def test_build_failures_redact_credentials_and_close_client(build_environment, monkeypatch, submission_failure):
    module, client = build_environment
    reply = (
        response("denied private-password", status=403)
        if submission_failure
        else response({"job_name": "cloud-job", "status": "failure", "details": "failed private-password"})
    )
    monkeypatch.setattr(image_build, "request", AsyncMock(return_value=reply))
    with pytest.raises(click.ClickException):
        await image_build.build_image(
            module_root=module,
            job_name=None if submission_failure else "cloud-job",
            metadata={"runtime": "pycharm-mcp"},
        )
    record = module.parent / ".runtime" / module.name / ("build-error.txt" if submission_failure else "image.json")
    assert "private-password" not in record.read_text()
    assert "[REDACTED]" in record.read_text()
    client.close.assert_awaited_once()


def test_module_build_commands_use_cloud_and_openenv_has_no_builder(monkeypatch):
    from click.testing import CliRunner

    from experiments.ide_mcp import cli as ide_cli
    from experiments.module_3_openenv_bash.cli import cli as openenv_cli

    monkeypatch.setattr(ide_cli, "configure", lambda *_: None)
    cloud = AsyncMock()
    monkeypatch.setattr(ide_cli, "build_cloud_image", cloud)
    command = ide_cli.create_cli(Path("module_4_jetbrains_mcp"), "bundled")
    result = CliRunner().invoke(command, ["build-image", "--job", "existing-job"])
    assert result.exit_code == 0, result.output
    assert cloud.await_args.kwargs["job_name"] == "existing-job"
    assert cloud.await_args.kwargs["metadata"] == {"runtime": "pycharm-mcp"}
    assert "build-image" not in openenv_cli.commands
