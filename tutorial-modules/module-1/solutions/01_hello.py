"""The finished ``01_hello.py``: start one sandbox, run bash and Python in it, delete it.

uv run solutions/01_hello.py
"""

import asyncio
import logging
import os

import structlog
from dotenv import load_dotenv
from idegym.client.client import IdeGYMClient, ServerCloseAction

load_dotenv()
# The IdeGYM client logs through structlog, which prints every level unless configured.
structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.INFO))

ORCHESTRATOR_URL = os.environ["IDEGYM_ORCHESTRATOR_URL"]
NAMESPACE = os.environ.get("IDEGYM_NAMESPACE", "idegym")
CLIENT_NAME = os.environ.get("IDEGYM_CLIENT_NAME", "ml-in-pl")

IMAGE = "ghcr.io/jetbrains-research/idegym/server-debian-bookworm-20250520-slim:0.13.0"


async def main() -> None:
    async with IdeGYMClient(orchestrator_url=ORCHESTRATOR_URL, name=CLIENT_NAME, namespace=NAMESPACE) as client:
        # STOP deletes the pod when the block exits, even on an exception.
        async with client.with_server(
            image_tag=IMAGE,
            server_name="hello",
            close_action=ServerCloseAction.STOP,
        ) as server:
            print(f"server {server.server_id} is up")

            result = await server.execute_bash(script="uname -a && whoami", command_timeout=30)
            print(result.stdout)

            result = await server.execute_bash(script="/opt/idegym/.venv/bin/python -c 'print(6 * 7)'")
            print(f"python in the sandbox says: {result.stdout.strip()}")


if __name__ == "__main__":
    asyncio.run(main())
