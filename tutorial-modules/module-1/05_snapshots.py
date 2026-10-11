"""Snapshot a running sandbox, then start a second sandbox from that snapshot.

Server #1 writes a file and is snapshotted, memory and disk included, then deleted. Server #2
starts from the snapshot instead of the image, so the file is already there. This is how a task
can be prepared once (a cloned repo, an indexed IDE) and then handed out many times.

Pod snapshots are GKE-only: they need gVisor and a non-E2 node, and a restore must use exactly the
same server configuration as the snapshot. That is why both servers share ``SERVER``.

    uv run 05_snapshots.py
"""

import asyncio
import logging
import os

import structlog
from dotenv import load_dotenv
from idegym.api.cpu import CpuQuantity
from idegym.api.memory import MemoryQuantity
from idegym.api.orchestrator.servers import ServerReuseStrategy, SnapshotRef
from idegym.api.resources import KubernetesResources, ResourceQuantities
from idegym.client.client import IdeGYMClient, ServerCloseAction
from idegym.client.operations.utils import PollingConfig

load_dotenv()
structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.INFO))

IMAGE = "ghcr.io/jetbrains-research/idegym/server-debian-bookworm-20250520-slim:0.13.0"
FILE_PATH = "/tmp/hello.txt"

SERVER = dict(
    image_tag=IMAGE,
    server_name="snapshot-demo",
    runtime_class_name="gvisor",
    run_as_root=True,
    resources=KubernetesResources(
        requests=ResourceQuantities(cpu=CpuQuantity(millicores=250), memory=MemoryQuantity(mi=512)),
        limits=ResourceQuantities(cpu=CpuQuantity(millicores=500), memory=MemoryQuantity(gi=1)),
    ),
    # GKE pod snapshots do not support E2 machines, and a restore must land on the same family.
    node_selector={"cloud.google.com/machine-family": "n2"},
    # NONE: always a new pod, so server #2 really comes from the snapshot.
    reuse_strategy=ServerReuseStrategy.NONE,
    close_action=ServerCloseAction.STOP,
)


async def main() -> None:
    async with IdeGYMClient(
        orchestrator_url=os.environ["IDEGYM_ORCHESTRATOR_URL"],
        name=os.environ.get("IDEGYM_CLIENT_NAME", "ml-in-pl"),
        namespace=os.environ.get("IDEGYM_NAMESPACE", "idegym"),
    ) as client:
        async with client.with_server(**SERVER) as server:
            await server.create_file(file_path=FILE_PATH, content="written by server #1\n")
            print(f"[server #1] {server.server_id} wrote {FILE_PATH}")

            # GKE can take up to two minutes to checkpoint the pod.
            snapshot = await server.snapshot(polling_config=PollingConfig(wait_timeout_in_sec=300))
            print(f"[server #1] snapshot {snapshot.snapshot_id} ({snapshot.snapshot_tag})")
        print("[server #1] deleted")

        async with client.with_server(
            **SERVER, snapshot=SnapshotRef(id=snapshot.snapshot_id, tag=snapshot.snapshot_tag)
        ) as server:
            result = await server.execute_bash(script=f"cat {FILE_PATH}")
            print(f"[server #2] {server.server_id} reads: {result.stdout.strip()}")


if __name__ == "__main__":
    asyncio.run(main())
