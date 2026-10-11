"""Run the actual upstream TB2 local backend and official client, without mocks.

Run with `uv run --locked --extra openenv python tests/openenv_workshop_smoke.py`.
Temporary paths replace TB2's container-root paths so this smoke check can run
on a developer machine.
"""

import asyncio
import json
import os
import socket
import sys
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import patch

import uvicorn
from openenv.core.env_server import create_app
from tbench2_env import Tbench2Action, Tbench2Observation
from tbench2_env.server import tbench2_env_environment as upstream

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.module_3_openenv_bash.task import (
    PROJECT_PATH,
    checks_from_output,
    reference_command,
    runtime_setup,
    verifier_setup,
)
from experiments.module_3_openenv_bash.workshop import run_client


async def check_client(url: str, task: Path, project: Path) -> None:
    results = []
    bootstrap = task.parent / "bootstrap"
    bootstrap.mkdir(exist_ok=True)
    tests, logs = task.parent / "staged-tests", task.parent / "verifier-logs"
    setup = runtime_setup(str(project), str(task))
    verifier = (
        verifier_setup(str(project), str(task))
        .replace("/logs/verifier", str(logs))
        .replace("/tests/verify.py", str(tests / "verify.py"))
    )
    for actions in [
        [{"action_type": "exec", "command": "pwd; test ! -e " + str(task / "tests")}],
        [
            {"action_type": "exec", "command": reference_command().replace(PROJECT_PATH, str(project))},
        ],
    ]:
        actions += [{"action_type": "exec", "command": verifier}, {"action_type": "evaluate"}]
        requests = iter(
            [*[json.dumps({"op": "step", "action": action}) for action in actions], json.dumps({"op": "state"}), ""]
        )
        await run_client(
            url,
            {"task_path": str(task), "task_id": "calculator-rename"},
            lambda: next(requests),
            results.append,
            setup_command=setup,
            bootstrap_path=str(bootstrap),
        )
    assert len(results) == 10
    assert [tool["name"] for tool in results[0]["tools"]] == ["bash"]
    assert results[1]["observation"]["success"]
    assert str(project) in results[1]["observation"]["output"]
    assert results[3]["done"] and results[3]["reward"] == 0
    assert checks_from_output(results[3]["observation"]["output"]).reward == 0
    assert results[6]["observation"]["success"]
    assert results[8]["done"] and results[8]["reward"] == 1
    assert checks_from_output(results[8]["observation"]["output"]).reward == 1
    assert results[9]["last_action_type"] == "evaluate"


def main() -> None:
    os.environ["IDEGYM_AUTH_USERNAME"] = "local-test"
    os.environ["IDEGYM_AUTH_PASSWORD"] = "local-test"
    os.environ["ENABLE_WEB_INTERFACE"] = "false"
    os.environ["CAMEL_RUNTIME"] = "true"
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        task, project = root / "task", root / "project"
        tests, logs = root / "staged-tests", root / "verifier-logs"
        os.environ["TB2_TASK_WORKDIR"] = str(project)
        with (
            patch.object(upstream, "_VERIFY_TESTS_DIR", str(tests)),
            patch.object(upstream, "_VERIFIER_LOG_DIR", str(logs)),
        ):
            app = create_app(
                lambda: upstream.Tbench2Environment(output_dir=str(root / "output")),
                Tbench2Action,
                Tbench2Observation,
                max_concurrent_envs=1,
            )
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                port = listener.getsockname()[1]
                server = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="on", ws="websockets"))
                thread = threading.Thread(target=lambda: server.run(sockets=[listener]), daemon=True)
                thread.start()
                try:
                    deadline = time.monotonic() + 20
                    while not server.started and thread.is_alive() and time.monotonic() < deadline:
                        time.sleep(0.05)
                    assert server.started, "Upstream OpenEnv server did not start"
                    for withhold in (False, True):
                        upstream.Tbench2Environment._WITHHELD_TESTS.clear()
                        os.environ["TB2_WITHHOLD_TESTS"] = "1" if withhold else "0"
                        asyncio.run(check_client(f"http://127.0.0.1:{port}", task, project))
                finally:
                    server.should_exit = True
                    thread.join(timeout=10)
                assert not thread.is_alive(), "Upstream OpenEnv server did not stop"
            assert not tests.exists() and not logs.exists(), "Upstream verifier artifacts were not cleaned up"
    assert "nemo_gym" not in sys.modules, "OpenEnv worker should not load Gym services"
    print("Official Tbench2Env: local shell, baseline failure, reference repair, five checks, reward, cleanup: PASS")


if __name__ == "__main__":
    main()
