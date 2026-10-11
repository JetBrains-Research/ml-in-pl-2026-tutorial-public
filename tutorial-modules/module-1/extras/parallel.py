"""Run several sandboxes at once from one client, the way rollouts run in parallel.

Watch them come and go on the dashboard.

    uv run extras/parallel.py
"""

import asyncio
import logging
import os

import structlog
from dotenv import load_dotenv
from idegym.api.orchestrator.servers import ServerReuseStrategy
from idegym.client.client import IdeGYMClient, ServerCloseAction

load_dotenv()
# The IdeGYM client logs through structlog, which prints every level unless configured.
structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.INFO))

SANDBOXES = 3


async def main() -> None:
    async with IdeGYMClient(
        orchestrator_url=os.environ["IDEGYM_ORCHESTRATOR_URL"],
        name=os.environ.get("IDEGYM_CLIENT_NAME", "ml-in-pl"),
        namespace=os.environ.get("IDEGYM_NAMESPACE", "idegym"),
    ) as client:

        async def rollout(index: int) -> str:
            async with client.with_server(
                image_tag=os.environ["BASH_IMAGE"],
                server_name=f"parallel-{index}",
                # NONE: always a fresh pod, never one another rollout just released.
                reuse_strategy=ServerReuseStrategy.NONE,
                close_action=ServerCloseAction.STOP,
            ) as server:
                result = await server.execute_bash(script=f"sleep 5 && echo rollout {index} on $(hostname)")
                return result.stdout.strip()

        for line in await asyncio.gather(*(rollout(i) for i in range(SANDBOXES))):
            print(line)


if __name__ == "__main__":
    asyncio.run(main())
