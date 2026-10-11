"""IDEGym provisions the server image; an official client worker owns its session."""

import asyncio
import json
import os
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

from idegym.api.auth import BasicAuth
from idegym.api.config import OTELConfig, TracingConfig
from idegym.api.orchestrator.servers import ErrorResponse, ServerKind, ServerReuseStrategy
from idegym.api.resources import KubernetesResources
from idegym.client.client import IdeGYMClient
from idegym.client.operations.utils import PollingConfig
from pydantic import BaseModel, Field

from experiments.ide_mcp.transport import ToolInfo, ToolResult
from experiments.module_3_openenv_bash.task import reset_parameters


class TerminalSettings(BaseModel):
    image: str
    ready_timeout: int = Field(default=600, ge=1)


class OpenEnvBridge:
    def __init__(self, base_url: str, output: Path) -> None:
        self.base_url = base_url
        self.output = output
        self.process = None
        self.tools: list[ToolInfo] = []
        self.done = False
        self._lock = asyncio.Lock()
        self._log = None
        self.scope: dict[str, Any] = {}

    async def __aenter__(self) -> "OpenEnvBridge":
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env.pop("VIRTUAL_ENV", None)
        self._log = (self.output / "openenv-client.log").open("w")
        try:
            self.process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-u",
                str(Path(__file__).with_name("worker.py")),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=self._log,
                env=env,
            )
            initial = await self.request({"base_url": self.base_url, "reset": reset_parameters()})
            self.tools = [ToolInfo.model_validate(tool) for tool in initial["tools"]]
            required = {"bash"}
            if not required.issubset(tool.name for tool in self.tools):
                raise RuntimeError("The image does not advertise the expected terminal action interface")
            return self
        except BaseException:
            await self.__aexit__(None, None, None)
            raise

    async def request(self, payload: dict[str, Any], timeout: int = 180) -> dict[str, Any]:
        async with self._lock:
            self.process.stdin.write((json.dumps(payload) + "\n").encode())
            await self.process.stdin.drain()
            line = await asyncio.wait_for(self.process.stdout.readline(), timeout)
            if not line or not (response := json.loads(line)).get("ok"):
                raise RuntimeError("OpenEnv request failed; inspect openenv-client.log")
            with (self.output / "openenv-protocol.jsonl").open("a") as log:
                log.write(json.dumps({"request": payload, "response": response}) + "\n")
            return response["data"]

    async def discover(self) -> list[ToolInfo]:
        return self.tools

    async def call(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        if name != "bash":
            raise ValueError(f"Unsupported model tool: {name}")
        if set(arguments) != {"command"} or not isinstance(arguments["command"], str):
            return ToolResult(is_error=True, content=[{"type": "text", "text": "bash expects one string command"}])
        data = await self.request({"op": "step", "action": {"action_type": "exec", "command": arguments["command"]}})
        self.done = bool(data["done"])
        observation = data["observation"]
        return ToolResult(
            is_error=not observation["success"],
            content=[{"type": "text", "text": observation["output"] or observation["error"]}],
        )

    async def evaluate(self) -> dict[str, Any]:
        data = await self.request({"op": "step", "action": {"action_type": "evaluate"}})
        self.done = bool(data["done"])
        return data

    async def state(self) -> dict[str, Any]:
        return await self.request({"op": "state"})

    async def __aexit__(self, *args) -> None:
        try:
            if self.process and self.process.returncode is None:
                self.process.stdin.close()
                try:
                    await asyncio.wait_for(self.process.wait(), 10)
                except TimeoutError:
                    self.process.kill()
                    await self.process.wait()
        finally:
            if self._log:
                self._log.close()


@asynccontextmanager
async def connect_openenv(settings: TerminalSettings, output: Path) -> AsyncIterator[OpenEnvBridge]:
    polling = PollingConfig(wait_timeout_in_sec=settings.ready_timeout)
    async with IdeGYMClient(
        orchestrator_url=os.environ["IDEGYM_ORCHESTRATOR_URL"],
        auth=BasicAuth(username=os.environ["IDEGYM_AUTH_USERNAME"], password=os.environ["IDEGYM_AUTH_PASSWORD"]),
        name="ml-in-pl-openenv-bash",
        namespace=os.environ.get("IDEGYM_NAMESPACE", "idegym"),
        nodes_count=0,
        request_timeout_in_seconds=settings.ready_timeout,
        otel_config=OTELConfig(tracing=TracingConfig(endpoint=None)),
    ) as client:
        server = await client.start_server(
            image_tag=settings.image,
            server_name=f"ml-in-pl-openenv-bash-{uuid4().hex[:8]}",
            server_kind=ServerKind.OPENENV,
            service_port=8000,
            container_port=8000,
            # Official TB2 grades at /tests and /logs. gVisor isolates the
            # root-based upstream image while preserving those filesystem paths.
            run_as_root=True,
            runtime_class_name="gvisor",
            reuse_strategy=ServerReuseStrategy.NONE,
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
            async with OpenEnvBridge(server.openenv_url, output) as bridge:
                bridge.scope = {"server_id": server.server_id, "image": settings.image}
                yield bridge
        finally:
            stopped = await client.stop_server(server, polling_config=polling)
            if isinstance(stopped, ErrorResponse):
                raise RuntimeError("IDEGym failed to delete the OpenEnv server")
