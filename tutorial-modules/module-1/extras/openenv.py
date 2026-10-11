"""Reach the sandbox through OpenEnv, the protocol many RL frameworks already speak.

The image runs an OpenEnv bash environment next to the IdeGYM server, on port 8001. The official
``EnvClient`` talks to it over a WebSocket that the orchestrator forwards and authenticates;
``server.openenv_url`` is that WebSocket's base URL. An agent would send the same commands as in
``extras/bash_agent.py``, only through ``reset``/``step``.

    uv run extras/openenv.py
"""

import asyncio
import base64
import logging
import os

import structlog
from dotenv import load_dotenv
from idegym.api.orchestrator.servers import ServerKind
from idegym.client.client import IdeGYMClient, ServerCloseAction
from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient
from openenv.core.env_server.types import Action, Observation, State
from websockets.asyncio.client import connect

load_dotenv()
# The IdeGYM client logs through structlog, which prints every level unless configured.
structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.INFO))


# Must match the environment's models on the server side (see the image definition).
class BashAction(Action):
    command: str
    timeout: int = 30


class BashObservation(Observation):
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0


class BashEnv(EnvClient[BashAction, BashObservation, State]):
    async def connect(self) -> "BashEnv":
        # EnvClient does not take auth headers, but the orchestrator requires Basic auth.
        if self._ws is None:
            credentials = f"{os.environ['IDEGYM_AUTH_USERNAME']}:{os.environ['IDEGYM_AUTH_PASSWORD']}"
            headers = {"Authorization": "Basic " + base64.b64encode(credentials.encode()).decode()}
            self._ws = await connect(self._ws_url, additional_headers=headers, open_timeout=self._connect_timeout)
        return self

    def _step_payload(self, action: BashAction) -> dict:
        return action.model_dump()

    def _parse_result(self, payload: dict) -> StepResult[BashObservation]:
        observation = BashObservation.model_validate(payload["observation"])
        return StepResult(observation=observation, reward=payload.get("reward"), done=payload.get("done", False))

    def _parse_state(self, payload: dict) -> State:
        return State.model_validate(payload)


async def main() -> None:
    async with IdeGYMClient(
        orchestrator_url=os.environ["IDEGYM_ORCHESTRATOR_URL"],
        name=os.environ.get("IDEGYM_CLIENT_NAME", "ml-in-pl"),
        namespace=os.environ.get("IDEGYM_NAMESPACE", "idegym"),
    ) as client:
        # An OpenEnv server: the orchestrator routes to the environment's port, not IdeGYM's.
        async with client.with_server(
            image_tag=os.environ["BASH_IMAGE"],
            server_name="openenv",
            server_kind=ServerKind.OPENENV,
            service_port=8001,
            container_port=8001,
            close_action=ServerCloseAction.STOP,
        ) as server:
            async with BashEnv(base_url=server.openenv_url, message_timeout_s=150) as env:
                await env.reset()
                for command in ["pwd && ls", "python -m pytest -q | tail -1"]:
                    step = await env.step(BashAction(command=command))
                    print(f"$ {command}\n{step.observation.stdout}{step.observation.stderr}")


if __name__ == "__main__":
    asyncio.run(main())
