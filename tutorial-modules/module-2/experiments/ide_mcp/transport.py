"""Discover and call IDE tools through the orchestrator's authenticated MCP gateway."""

import asyncio
import json
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import timedelta
from time import monotonic
from typing import Any
from uuid import uuid4

import httpx
from idegym.api.auth import BasicAuth
from idegym.api.config import OTELConfig, TracingConfig
from idegym.api.orchestrator.servers import ErrorResponse, ServerReuseStrategy
from idegym.api.resources import KubernetesResources
from idegym.api.type import OCIImageName
from idegym.client.client import IdeGYMClient
from idegym.client.operations.utils import PollingConfig
from idegym.client.server import IdeGYMServer
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.exceptions import McpError
from mcp.types import CallToolResult
from pydantic import BaseModel, Field

from experiments.ide_mcp.task import Mode


class ToolInfo(BaseModel):
    name: str
    description: str | None = None
    input_schema: dict[str, Any]

    def function(self) -> dict[str, Any]:
        return {
            "type": "function",
            "name": self.name,
            "description": self.description or self.name,
            "parameters": self.input_schema,
            "strict": False,
        }


class ToolList(BaseModel):
    tools: list[ToolInfo]


class ToolResult(BaseModel):
    content: list[dict[str, Any]]
    is_error: bool = False


class IdeSettings(BaseModel):
    mode: Mode
    image: OCIImageName = Field(min_length=1)
    ready_timeout: int = Field(default=600, ge=1)
    tool_timeout: int = Field(default=120, ge=1)


def unpack_result(result: CallToolResult) -> dict[str, Any]:
    if result.isError:
        raise RuntimeError("Orchestrator MCP tool failed: " + str(result.content))
    if result.structuredContent is not None:
        return result.structuredContent
    texts = [block.text for block in result.content if block.type == "text"]
    payload = json.loads("\n".join(texts))
    if not isinstance(payload, dict):
        raise RuntimeError("Expected an object from the orchestrator MCP tool")
    return payload


def select_tools(tools: list[ToolInfo], mode: Mode) -> list[ToolInfo]:
    # Gateway file/bash tools would let the agent bypass the IDE comparison.
    return [
        tool for tool in tools if tool.name.startswith("pycharm_") and not tool.name.startswith("pycharm_steroid_")
    ]


class IdeConnection:
    def __init__(self, session: ClientSession, server: IdeGYMServer, client_id: str, settings: IdeSettings) -> None:
        self.session = session
        self.server = server
        self.scope = {"client_id": client_id, "server_id": server.server_id}
        self.settings = settings

    async def request(self, name: str, fields: dict[str, Any]) -> CallToolResult:
        return await self.session.call_tool(
            name,
            {"request": self.scope | fields},
            read_timeout_seconds=timedelta(seconds=self.settings.tool_timeout),
        )

    async def discover(self) -> list[ToolInfo]:
        deadline = monotonic() + self.settings.ready_timeout
        last_error = "No tools for the selected IDE mode"
        while monotonic() < deadline:
            try:
                payload = unpack_result(await self.request("list_server_mcp_tools", {}))
                tools = select_tools(ToolList.model_validate(payload).tools, self.settings.mode)
                if tools:
                    return tools
            except (McpError, RuntimeError) as exc:
                # The pod's HTTP readiness precedes the IDE's asynchronous startup.
                last_error = str(exc)
            await asyncio.sleep(5)
        raise TimeoutError(f"IDE MCP did not become ready: {last_error}")

    async def call(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        result = await self.request("call_server_mcp_tool", {"tool_name": name, "arguments": arguments})
        # FastMCP can lift an upstream IDE tool error into the outer MCP error envelope.
        if result.isError:
            return ToolResult(is_error=True, content=[block.model_dump(mode="json") for block in result.content])
        return ToolResult.model_validate(unpack_result(result))


@asynccontextmanager
async def connect_ide(settings: IdeSettings) -> AsyncIterator[IdeConnection]:
    url = os.environ["IDEGYM_ORCHESTRATOR_URL"].rstrip("/")
    if not url.startswith(("https://", "http://")):
        url = "https://" + url
    username, password = os.environ["IDEGYM_AUTH_USERNAME"], os.environ["IDEGYM_AUTH_PASSWORD"]
    namespace = os.environ.get("IDEGYM_NAMESPACE", "idegym")
    polling = PollingConfig(wait_timeout_in_sec=settings.ready_timeout)
    async with IdeGYMClient(
        orchestrator_url=url,
        auth=BasicAuth(username=username, password=password),
        name=f"ml-in-pl-ide-{settings.mode}",
        namespace=namespace,
        nodes_count=0,
        request_timeout_in_seconds=settings.ready_timeout,
        otel_config=OTELConfig(tracing=TracingConfig(endpoint=None)),
    ) as client:
        server = await client.start_server(
            image_tag=settings.image,
            server_name=f"ml-in-pl-{settings.mode}-{uuid4().hex[:8]}",
            run_as_root=False,
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
            async with (
                httpx.AsyncClient(auth=httpx.BasicAuth(username, password), timeout=settings.tool_timeout) as http,
                streamable_http_client(url + "/mcp/", http_client=http) as (read, write, _),
                ClientSession(read, write) as session,
            ):
                await session.initialize()
                yield IdeConnection(session, server, str(client.client_id), settings)
        finally:
            stopped = await client.stop_server(server, polling_config=polling)
            if isinstance(stopped, ErrorResponse):
                raise RuntimeError("IDEGym failed to delete the evaluation sandbox")
