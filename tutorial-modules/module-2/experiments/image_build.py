# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Submit and poll image builds through the selected IDEGym Cloud Build deployment."""

import asyncio
import json
import os
from pathlib import Path
from time import monotonic

import click
from aiohttp import BasicAuth, ClientTimeout
from idegym.api.orchestrator.build import BuildFromYamlResponse
from idegym.api.orchestrator.jobs import JobStatusResponse
from idegym.api.status import Status
from nemo_gym.global_config import GlobalConfigDictParserConfig
from nemo_gym.server_utils import get_global_aiohttp_client, get_response_json, request

from experiments.common import ROOT


async def build_image(*, module_root: Path, job_name: str | None, metadata: dict[str, str]) -> None:
    """Submit or resume a build; record the job before waiting so it is recoverable."""
    record_path = ROOT / ".runtime" / module_root.name / "image.json"
    client = get_global_aiohttp_client(
        GlobalConfigDictParserConfig(
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            initial_global_config_dict=GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
        )
    )
    url = os.environ["IDEGYM_ORCHESTRATOR_URL"].rstrip("/")
    if not url.startswith(("https://", "http://")):
        url = "https://" + url
    auth = BasicAuth(os.environ["IDEGYM_AUTH_USERNAME"], os.environ["IDEGYM_AUTH_PASSWORD"])
    record_path.parent.mkdir(parents=True, exist_ok=True)
    click.echo(f"IDEGym Cloud Build orchestrator: {url}")
    try:
        if job_name is None:
            response = await request(
                "POST",
                url + "/api/build-push-images",
                auth=auth,
                timeout=ClientTimeout(total=120),
                json={
                    "namespace": os.environ.get("IDEGYM_NAMESPACE", "idegym"),
                    "yaml_content": (module_root / "image.yaml").read_text(),
                },
            )
            if response.status != 200:
                details = await response.text()
                details = redact_credentials(details)
                error_path = record_path.with_name("build-error.txt")
                error_path.write_text(details)
                raise click.ClickException(
                    f"IDEGym build submission returned HTTP {response.status}; details: {error_path}"
                )
            submitted = BuildFromYamlResponse.model_validate(await get_response_json(response))
            if len(submitted.job_names) != 1:
                raise click.ClickException(f"Expected one build job, received {len(submitted.job_names)}")
            job_name = submitted.job_names[0]
        record = {}
        if record_path.exists():
            previous = json.loads(record_path.read_text())
            if previous.get("job_name") == job_name or job_name in previous.get("job_names", []):
                record = previous
        record.update(**metadata, job_name=job_name, status="IN_PROGRESS", orchestrator_url=url)
        record_path.write_text(json.dumps(record, indent=2) + "\n")
        click.echo(f"Build job: {job_name}; record: {record_path}")
        deadline = monotonic() + 1800
        while monotonic() < deadline:
            response = await request(
                "GET", url + "/api/jobs/status/" + job_name, auth=auth, timeout=ClientTimeout(total=60)
            )
            if response.status != 200:
                raise click.ClickException(f"IDEGym build status returned HTTP {response.status}; job: {job_name}")
            status = JobStatusResponse.model_validate(await get_response_json(response))
            if status.status == Status.SUCCESS and status.tag:
                record.update(status="SUCCESS", image=status.tag)
                record_path.write_text(json.dumps(record, indent=2) + "\n")
                click.echo(f"Built image: {status.tag}")
                return
            if status.status != Status.IN_PROGRESS:
                record.update(status=str(status.status), details=redact_credentials(status.details or ""))
                record_path.write_text(json.dumps(record, indent=2) + "\n")
                raise click.ClickException(f"Build failed; inspect {record_path}")
            click.echo("Image build is still running...")
            await asyncio.sleep(20)
        raise click.ClickException(f"Build still running after 30 minutes; resume with build-image --job {job_name}")
    finally:
        await client.close()


def redact_credentials(text: str) -> str:
    for key, value in os.environ.items():
        if value and any(word in key.upper() for word in ("KEY", "PASSWORD", "TOKEN", "SECRET", "USERNAME")):
            text = text.replace(value, "[REDACTED]")
    return text
