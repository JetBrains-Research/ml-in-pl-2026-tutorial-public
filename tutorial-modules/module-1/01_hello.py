import asyncio
import logging
import os

import structlog
from dotenv import load_dotenv
from idegym.client.client import IdeGYMClient, ServerCloseAction  # noqa: F401 (used once the TODOs are done)

load_dotenv()
structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.INFO))

ORCHESTRATOR_URL = os.environ["IDEGYM_ORCHESTRATOR_URL"]
NAMESPACE = os.environ.get("IDEGYM_NAMESPACE", "idegym")
CLIENT_NAME = os.environ.get("IDEGYM_CLIENT_NAME", "ml-in-pl")

IMAGE = "ghcr.io/jetbrains-research/idegym/server-debian-bookworm-20250520-slim:0.13.0"


async def main() -> None:
    # TODO 1: open a client:
    #   async with IdeGYMClient(orchestrator_url=..., name=..., namespace=...) as client:

    # TODO 2: start a server from IMAGE, deleted when the block exits:
    #   async with client.with_server(image_tag=..., server_name="hello",
    #                                 close_action=ServerCloseAction.STOP) as server:

    # TODO 3: run `uname -a && whoami` with server.execute_bash(script=...) and print result.stdout

    # TODO 4: run Python in the sandbox: /opt/idegym/.venv/bin/python -c 'print(6 * 7)'
    ...


if __name__ == "__main__":
    asyncio.run(main())
