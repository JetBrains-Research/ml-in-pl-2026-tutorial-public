import asyncio
import logging
import os
import tempfile
from pathlib import Path

import structlog
from dotenv import load_dotenv
from idegym.client.client import IdeGYMClient, ServerCloseAction

load_dotenv()
structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.INFO))

IMAGE = "ghcr.io/jetbrains-research/idegym/server-debian-bookworm-20250520-slim:0.13.0"
PYTHON = "/opt/idegym/.venv/bin/python"


async def main() -> None:
    async with IdeGYMClient(
        orchestrator_url=os.environ["IDEGYM_ORCHESTRATOR_URL"],
        name=os.environ.get("IDEGYM_CLIENT_NAME", "ml-in-pl"),
        namespace=os.environ.get("IDEGYM_NAMESPACE", "idegym"),
    ) as client:
        async with client.with_server(
            image_tag=IMAGE,
            server_name="files",
            close_action=ServerCloseAction.STOP,
        ) as server:
            await server.create_file(file_path="/tmp/greet.py", content="name = 'world'\nprint(f'hello, {name}')\n")
            result = await server.execute_bash(script=f"{PYTHON} /tmp/greet.py", command_timeout=30)
            print(result.stdout.strip())

            # Replace line 1 only; the rest of the file stays as it is.
            await server.edit_file(
                file_path="/tmp/greet.py", start_line=1, end_line=1, new_content="name = 'ML in PL'"
            )
            result = await server.execute_bash(script=f"{PYTHON} /tmp/greet.py", command_timeout=30)
            print(result.stdout.strip())

            with tempfile.TemporaryDirectory() as directory:
                local = Path(directory) / "greet.py"
                await server.download_file(file_path="/tmp/greet.py", local_path=local)
                print(f"downloaded:\n{local.read_text()}")

                # Byte-for-byte upload, for data that is not text.
                local.write_bytes(bytes(range(256)))
                await server.upload_file(local_path=local, file_path="/tmp/blob.bin")
                result = await server.execute_bash(script="wc -c /tmp/blob.bin", command_timeout=30)
                print(result.stdout.strip())


if __name__ == "__main__":
    asyncio.run(main())
