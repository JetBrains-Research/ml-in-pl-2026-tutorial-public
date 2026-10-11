"""Reuse the bounded Gym tool loop and calculator verifier for non-MCP bash."""

import asyncio
import os
from pathlib import Path
from typing import Literal

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import get_response_json, raise_for_status
from pydantic import ConfigDict, Field, PrivateAttr

from experiments.bash_access.transport import BashMode, BashSettings
from experiments.ide_mcp.task import Grade


class BashAgentConfig(BaseResponsesAPIAgentConfig):
    model_server: ModelServerRef
    shell: BashSettings
    step_limit: int = Field(default=20, ge=1, le=100)


class BashRunRequest(BaseRunRequest):
    # Gym adds rollout indices used to correlate model-call captures.
    model_config = ConfigDict(extra="allow")

    task_id: Literal["calculator-rename"] = "calculator-rename"


class BashVerifyResponse(BaseVerifyResponse):
    task_id: str
    checks: Grade
    mode: BashMode
    golden: bool = False
    tool_calls: int
    successful_tool_calls: int


class BashAgent(SimpleResponsesAPIAgent):
    config: BashAgentConfig
    _semaphore: asyncio.Semaphore = PrivateAttr(default_factory=lambda: asyncio.Semaphore(1))

    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming) -> NeMoGymResponse:
        raise NotImplementedError("Use /run: bash tools require a scoped evaluation sandbox")

    async def run(self, body: BashRunRequest) -> BashVerifyResponse:
        from experiments.module_2_direct_bash.workshop import evaluate_bash

        async def model_call(params: NeMoGymResponseCreateParamsNonStreaming) -> NeMoGymResponse:
            reply = await self.server_client.post(
                server_name=self.config.model_server.name,
                url_path=self.url_path_for_run("/v1/responses", body),
                json=params,
            )
            await raise_for_status(reply)
            return NeMoGymResponse.model_validate(await get_response_json(reply))

        async with self._semaphore:
            return await evaluate_bash(
                body, self.config.shell, model_call, self.config.step_limit, Path(os.environ["POC_RUN_DIR"])
            )
