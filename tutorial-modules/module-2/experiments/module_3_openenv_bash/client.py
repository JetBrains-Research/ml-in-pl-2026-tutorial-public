"""IDEGym authentication for the official Terminal-Bench OpenEnv client."""

import asyncio
import base64
import os

from tbench2_env import Tbench2Env
from websockets.asyncio.client import connect


class WorkshopTerminalEnv(Tbench2Env):
    async def _connect_async(self) -> "WorkshopTerminalEnv":
        if self._ws is None:
            # The official client has no auth-header constructor option. Actions,
            # observations, reset, and state retain their upstream implementation.
            credentials = f"{os.environ['IDEGYM_AUTH_USERNAME']}:{os.environ['IDEGYM_AUTH_PASSWORD']}"
            authorization = "Basic " + base64.b64encode(credentials.encode()).decode()
            self._ws = await connect(
                self._ws_url,
                additional_headers={"Authorization": authorization},
                open_timeout=self._connect_timeout,
                max_size=self._max_message_size,
            )
            self._ws_loop = asyncio.get_running_loop()
        return self
