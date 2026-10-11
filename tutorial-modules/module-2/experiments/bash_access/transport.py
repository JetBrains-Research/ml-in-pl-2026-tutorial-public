"""Direct IDEGym shell transport for Module 2."""

import asyncio
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from httpx import HTTPStatusError
from idegym.api.auth import BasicAuth
from idegym.api.config import OTELConfig, TracingConfig
from idegym.api.orchestrator.servers import ErrorResponse, ServerKind, ServerReuseStrategy
from idegym.api.resources import KubernetesResources
from idegym.api.tools.bash import BashCommandResponse
from idegym.api.type import OCIImageName
from idegym.client.client import IdeGYMClient
from idegym.client.operations.utils import PollingConfig
from idegym.client.server import IdeGYMServer
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from experiments.calculator import setup_script
from experiments.ide_mcp.task import PROJECT_PATH, VERIFY_SCRIPT, Grade, parse_grade, python_command
from experiments.ide_mcp.transport import ToolInfo, ToolResult

BashMode = Literal["bash"]


class BashSettings(BaseModel):
    mode: BashMode
    image: OCIImageName
    ready_timeout: int = Field(default=600, ge=1)
    tool_timeout: int = Field(default=120, ge=1)


class BashArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")
    command: str = Field(min_length=1, description="Bash script to execute in /home/appuser/work.")
    timeout: int = Field(default=30, ge=1, le=120, description="Command timeout in seconds, at most 120.")


async def wait_for_forwarding(server: IdeGYMServer, timeout: float = 60) -> None:
    # A healthy pod can precede the gateway route. Probe with GET before any
    # task mutation; never retry model commands that may already have executed.
    try:
        async with asyncio.timeout(timeout):
            while True:
                try:
                    await server.forward("GET", "health", request_timeout=10)
                    return
                except RuntimeError as error:
                    cause = error.__context__
                    if not isinstance(cause, HTTPStatusError) or cause.response.status_code not in (502, 503, 504):
                        raise
                    await asyncio.sleep(1)
    except TimeoutError as error:
        raise RuntimeError(f"IDEGym forwarding readiness did not complete within {timeout}s") from error


class BashConnection:
    def __init__(self, server: IdeGYMServer) -> None:
        self.server = server

    async def execute_bash(self, script: str, command_timeout: int = 30) -> BashCommandResponse:
        from experiments.module_2_direct_bash.workshop import execute_direct

        return await execute_direct(self.server, script, command_timeout)

    async def discover(self) -> list[ToolInfo]:
        return [
            ToolInfo(
                name="bash",
                description=(
                    "Execute bash in the remote task sandbox. Files persist between calls; "
                    "shell variables and cwd changes do not. Python and pytest are on PATH. "
                    "Returns stdout, stderr, and exit_code."
                ),
                input_schema=BashArguments.model_json_schema(),
            )
        ]

    async def prepare(self, tools: list[ToolInfo]) -> None:
        # The generic Bookworm image contains no task files. Setup belongs to the runner.
        staged = await self.server.execute_bash(script=python_command(setup_script(PROJECT_PATH)), command_timeout=30)
        if staged.exit_code:
            raise RuntimeError("Could not stage the calculator project")
        dependencies = await self.execute_bash(
            "uv pip install --python /opt/idegym/.venv/bin/python pytest==8.4.2", command_timeout=120
        )
        if dependencies.exit_code:
            raise RuntimeError("Could not install the calculator's pytest dependency")
        result = await self.execute_bash(python_command(VERIFY_SCRIPT))
        checks = parse_grade(result.stdout, result.exit_code)
        expected = Grade(
            renamed=False, references_updated=False, arithmetic=False, zero_division=True, main_output=True
        )
        if checks != expected:
            raise RuntimeError("Staged calculator does not match the expected unmodified task")

    async def call(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        try:
            if name != "bash":
                raise ValueError("Only bash is exposed")
            action = BashArguments.model_validate(arguments)
        except (ValidationError, ValueError) as exc:
            return ToolResult(is_error=True, content=[{"type": "text", "text": str(exc)}])
        result = await self.execute_bash(action.command, action.timeout)
        return ToolResult(is_error=result.exit_code != 0, content=[{"type": "text", "text": result.model_dump_json()}])


@asynccontextmanager
async def connect_bash(settings: BashSettings, output: Path) -> AsyncIterator[BashConnection]:
    url = os.environ["IDEGYM_ORCHESTRATOR_URL"].rstrip("/")
    if not url.startswith(("https://", "http://")):
        url = "https://" + url
    polling = PollingConfig(wait_timeout_in_sec=settings.ready_timeout)
    async with IdeGYMClient(
        orchestrator_url=url,
        auth=BasicAuth(username=os.environ["IDEGYM_AUTH_USERNAME"], password=os.environ["IDEGYM_AUTH_PASSWORD"]),
        name=f"ml-in-pl-{settings.mode}",
        namespace=os.environ.get("IDEGYM_NAMESPACE", "idegym"),
        nodes_count=0,
        request_timeout_in_seconds=settings.ready_timeout,
        otel_config=OTELConfig(tracing=TracingConfig(endpoint=None)),
    ) as client:
        server = await client.start_server(
            image_tag=settings.image,
            server_name=f"ml-in-pl-{settings.mode}-{uuid4().hex[:8]}",
            run_as_root=False,
            reuse_strategy=ServerReuseStrategy.NONE,
            server_kind=ServerKind.IDEGYM,
            service_port=80,
            container_port=8000,
            server_start_wait_timeout_in_seconds=settings.ready_timeout,
            polling_config=polling,
            resources=KubernetesResources.model_validate(
                {
                    "requests": {"cpu": "1", "memory": "4Gi", "ephemeral-storage": "12Gi"},
                    "limits": {"cpu": "2", "memory": "8Gi", "ephemeral-storage": "20Gi"},
                }
            ),
        )
        try:
            await wait_for_forwarding(server, timeout=min(settings.ready_timeout, 60))
            yield BashConnection(server)
        finally:
            stopped = await client.stop_server(server, polling_config=polling)
            if isinstance(stopped, ErrorResponse):
                raise RuntimeError("IdeGYM failed to delete the evaluation sandbox")
