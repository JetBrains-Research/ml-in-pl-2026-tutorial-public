"""NeMo Gym adapter for the official OpenEnv terminal episode."""

import asyncio
import os
from pathlib import Path

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import get_response_json, raise_for_status
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from experiments.ide_mcp.task import Grade
from experiments.module_3_openenv_bash.transport import TerminalSettings


class TerminalAgentConfig(BaseResponsesAPIAgentConfig):
    model_server: ModelServerRef
    environment: TerminalSettings
    step_limit: int = Field(default=20, ge=1, le=100)


class TerminalRunRequest(BaseRunRequest):
    # Gym adds rollout indices used to correlate model-call captures.
    model_config = ConfigDict(extra="allow")

    task_id: str = "calculator-rename"


class TerminalEvaluationResult(BaseModel):
    reward: float
    task_id: str = "calculator-rename"
    checks: Grade
    mode: str = "openenv"
    golden: bool = False
    tool_calls: int
    successful_tool_calls: int
    environment_reward: float


class TerminalResult(BaseVerifyResponse, TerminalEvaluationResult):
    pass


class TerminalAgent(SimpleResponsesAPIAgent):
    config: TerminalAgentConfig
    _semaphore: asyncio.Semaphore = PrivateAttr(default_factory=lambda: asyncio.Semaphore(1))

    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming) -> NeMoGymResponse:
        raise NotImplementedError("Use /run: OpenEnv tools require a scoped environment session")

    async def run(self, body: TerminalRunRequest) -> TerminalResult:
        from experiments.module_3_openenv_bash.workshop import evaluate

        async def model_call(params: NeMoGymResponseCreateParamsNonStreaming) -> NeMoGymResponse:
            reply = await self.server_client.post(
                server_name=self.config.model_server.name,
                url_path=self.url_path_for_run("/v1/responses", body),
                json=params,
            )
            await raise_for_status(reply)
            return NeMoGymResponse.model_validate(await get_response_json(reply))

        async with self._semaphore:
            return await evaluate(
                self.config.environment,
                Path(os.environ["POC_RUN_DIR"]),
                body=body,
                model_call=model_call,
                step_limit=self.config.step_limit,
            )
