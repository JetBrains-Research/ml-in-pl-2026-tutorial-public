"""Gym agent for repairing diagnostics from PyCharm's integrated MCP."""

import asyncio
import os
from pathlib import Path
from typing import Literal

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import SimpleResponsesAPIAgent
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import get_response_json, raise_for_status
from pydantic import ConfigDict, PrivateAttr

from experiments.ide_mcp.agent import IdeAgentConfig
from experiments.module_4_jetbrains_mcp.evaluation import InspectionResult
from experiments.module_4_jetbrains_mcp.workshop import evaluate


class InspectionRunRequest(BaseRunRequest):
    # Gym adds rollout indices used to correlate model-call captures.
    model_config = ConfigDict(extra="allow")

    task_id: Literal["ide-inspections"] = "ide-inspections"


class InspectionVerifyResponse(BaseVerifyResponse):
    inspection_result: InspectionResult


class InspectionAgent(SimpleResponsesAPIAgent):
    config: IdeAgentConfig
    _semaphore: asyncio.Semaphore = PrivateAttr(default_factory=lambda: asyncio.Semaphore(1))

    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming) -> NeMoGymResponse:
        raise NotImplementedError("Use /run for scoped inspection evaluation")

    async def run(self, body: InspectionRunRequest) -> InspectionVerifyResponse:
        async def model_call(params: NeMoGymResponseCreateParamsNonStreaming) -> NeMoGymResponse:
            reply = await self.server_client.post(
                server_name=self.config.model_server.name,
                url_path=self.url_path_for_run("/v1/responses", body),
                json=params,
            )
            await raise_for_status(reply)
            return NeMoGymResponse.model_validate(await get_response_json(reply))

        async with self._semaphore:
            result, response = await evaluate(
                self.config.ide,
                Path(os.environ["POC_RUN_DIR"]),
                params=body.responses_create_params,
                model_call=model_call,
                step_limit=self.config.step_limit,
            )
        return InspectionVerifyResponse(
            **body.model_dump(),
            response=response,
            reward=result.reward,
            inspection_result=result,
        )
