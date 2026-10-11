"""Build a sandbox image from plain Debian on the cluster, then start a sandbox from it.

The image is described in Python with plugins: system packages, a user, the IdeGYM server, and
a project (here the IdeGYM repository itself). The orchestrator turns the description into a
Dockerfile, builds it in the cluster, and pushes it to its registry; nothing is built locally.
Expect the build to take several minutes.

    uv run 04_build_image.py
"""

import asyncio
import logging
import os
import tempfile
from pathlib import Path

import structlog
from dotenv import load_dotenv
from idegym.client.client import IdeGYMClient, ServerCloseAction
from idegym.image.builder import Image
from idegym.plugins.defaults.image import BaseSystem, IdeGYMServer, Project, User

load_dotenv()
structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.INFO))

IDEGYM_REPO = "https://github.com/JetBrains-Research/idegym.git"
IDEGYM_REF = "v0.13.0"


async def main() -> None:
    image = (
        Image.from_base("debian:bookworm-20250520-slim")
        .named("ml-in-pl-idegym-project-")
        .with_plugin(BaseSystem())
        .with_plugin(User(username="appuser", uid=1000, gid=1000))
        .with_plugin(IdeGYMServer.from_git(url=IDEGYM_REPO, ref=IDEGYM_REF))
        .with_plugin(Project.from_git(url=IDEGYM_REPO, ref=IDEGYM_REF, owner="appuser", target="/home/appuser/work"))
    )
    print(image.to_spec().dockerfile_content)

    async with IdeGYMClient(
        orchestrator_url=os.environ["IDEGYM_ORCHESTRATOR_URL"],
        name=os.environ.get("IDEGYM_CLIENT_NAME", "ml-in-pl"),
        namespace=os.environ.get("IDEGYM_NAMESPACE", "idegym"),
    ) as client:
        # The orchestrator takes the YAML form of the image, not the Python object.
        with tempfile.TemporaryDirectory() as directory:
            definition = image.write_yaml(Path(directory) / "image.yaml")
            summary = await client.build_and_push_images(path=definition, timeout=3600)
        job = summary.jobs_results[0]
        if summary.failed_jobs:
            raise RuntimeError(f"Build {job.job_name} failed: {job.details}")
        print(f"built {job.tag} in {summary.total_time}")

        async with client.with_server(
            image_tag=job.tag,
            server_name="custom-image",
            close_action=ServerCloseAction.STOP,
        ) as server:
            # Plain Debian has no Python; the server plugin brought its own under /opt/idegym.
            result = await server.execute_bash(
                script="whoami && ls /home/appuser/work && /opt/idegym/.venv/bin/python --version"
            )
            print(result.stdout)


if __name__ == "__main__":
    asyncio.run(main())
