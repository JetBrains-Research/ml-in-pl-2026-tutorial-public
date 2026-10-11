"""Verify a supplied generic TB2 image using its real client; no image build."""

import asyncio
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from uuid import uuid4

import click
from tbench2_env import Tbench2Action

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.module_3_openenv_bash.client import WorkshopTerminalEnv
from experiments.module_3_openenv_bash.task import (
    checks_from_output,
    reference_command,
    reset_parameters,
    runtime_setup,
    verifier_setup,
)


async def check_client(url: str) -> None:
    async with WorkshopTerminalEnv(base_url=url, message_timeout_s=150) as client:
        await client.reset(task_id="workshop-bootstrap", task_path="/tmp")
        staged = await client.step(Tbench2Action(command=runtime_setup()))
        assert staged.observation.success, staged.observation.error
        await client.reset(**reset_parameters())
        probe = await client.step(Tbench2Action(command="pwd; python3 main.py"))
        assert probe.observation.success, probe.observation.error
        assert "15 / 3 = 5.0" in probe.observation.output, probe.observation.output
        assert (await client.step(Tbench2Action(command=verifier_setup()))).observation.success
        baseline = await client.step(Tbench2Action(action_type="evaluate"))
        assert baseline.done and baseline.reward == 0, baseline
        assert checks_from_output(baseline.observation.output).reward == 0
    async with WorkshopTerminalEnv(base_url=url, message_timeout_s=150) as client:
        await client.reset(task_id="workshop-bootstrap", task_path="/tmp")
        staged = await client.step(Tbench2Action(command=runtime_setup()))
        assert staged.observation.success, staged.observation.error
        await client.reset(**reset_parameters())
        await client.step(Tbench2Action(command=reference_command()))
        assert (await client.step(Tbench2Action(command=verifier_setup()))).observation.success
        result = await client.step(Tbench2Action(action_type="evaluate"))
        assert result.done and result.reward == 1, result
        assert checks_from_output(result.observation.output).reward == 1
        assert (await client.state()).last_action_type == "evaluate"


@click.command()
@click.option("--image", required=True, help="Supplied official tbench2_env server image.")
def main(image: str) -> None:
    """Start the image, check native shell execution and grading, then remove it."""
    name = "ml-in-pl-openenv-smoke-" + uuid4().hex[:8]
    os.environ["IDEGYM_AUTH_USERNAME"] = "local-test"
    os.environ["IDEGYM_AUTH_PASSWORD"] = "local-test"
    try:
        subprocess.run(
            [
                "docker",
                "run",
                "--detach",
                "--rm",
                "--platform",
                "linux/amd64",
                "--name",
                name,
                "--publish",
                "127.0.0.1::8000",
                image,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            timeout=180,
        )
        ports = json.loads(
            subprocess.check_output(
                ["docker", "inspect", name, "--format", "{{json .NetworkSettings.Ports}}"],
                text=True,
            )
        )
        url = "http://127.0.0.1:" + ports["8000/tcp"][0]["HostPort"]
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(url + "/health", timeout=2) as response:
                    if response.status == 200:
                        break
            except (OSError, TimeoutError):
                time.sleep(0.5)
        else:
            raise RuntimeError("Supplied image did not become healthy")
        asyncio.run(check_client(url))
        subprocess.run(
            [
                "docker",
                "exec",
                name,
                "python",
                "-c",
                "from pathlib import Path; assert not Path('/tests').exists(); "
                "assert not Path('/logs/verifier').exists()",
            ],
            check=True,
            timeout=20,
        )
        click.echo("Supplied image: runtime setup, baseline reward 0, reference reward 1, five checks, cleanup: PASS")
    except Exception:
        subprocess.run(["docker", "logs", "--tail", "40", name], timeout=20)
        raise
    finally:
        subprocess.run(
            ["docker", "rm", "--force", name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
        )


if __name__ == "__main__":
    main()
