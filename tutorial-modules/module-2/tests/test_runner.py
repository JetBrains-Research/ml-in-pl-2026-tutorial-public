# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import sys
from pathlib import Path

import pytest

from experiments.common import run_logged


def test_stalled_runner_times_out_and_stops_driver(tmp_path: Path) -> None:
    import subprocess

    marker = tmp_path / "terminated"
    driver = (
        "import signal, sys, time; from pathlib import Path; "
        f"signal.signal(signal.SIGTERM, lambda *_: (Path({str(marker)!r}).touch(), sys.exit(0))); "
        "time.sleep(60)"
    )
    with (tmp_path / "run.log").open("w") as log:
        with pytest.raises(subprocess.TimeoutExpired):
            run_logged([sys.executable, "-c", driver], log, timeout_s=0.5)
    assert marker.exists()


def test_runner_cleans_up_descendant_after_driver_exits(tmp_path: Path) -> None:
    """Exercise the shell-child leak observed in the Gym branch's shutdown."""
    marker = tmp_path / "terminated"
    ready = tmp_path / "ready"
    child = (
        "import signal, sys, time; from pathlib import Path; "
        f"signal.signal(signal.SIGTERM, lambda *_: (Path({str(marker)!r}).write_text('stopped'), sys.exit(0))); "
        f"Path({str(ready)!r}).touch(); time.sleep(60)"
    )
    driver = (
        "import subprocess, sys, time; from pathlib import Path; "
        f"subprocess.Popen([sys.executable, '-c', {child!r}]); "
        "deadline = time.monotonic() + 5\n"
        f"while not Path({str(ready)!r}).exists() and time.monotonic() < deadline: time.sleep(0.05)\n"
        f"assert Path({str(ready)!r}).exists()"
    )
    with (tmp_path / "child.log").open("w") as log:
        assert run_logged([sys.executable, "-c", driver], log) == 0
    assert marker.read_text() == "stopped"


def test_completed_driver_preserves_exit_status_when_group_probe_denied(monkeypatch, tmp_path):
    from unittest.mock import Mock

    import pytest

    from experiments import common

    process = Mock(pid=12345)
    process.wait.return_value = 0
    process.poll.return_value = 0
    monkeypatch.setattr(common.subprocess, "Popen", Mock(return_value=process))
    monkeypatch.setattr(common.os, "killpg", Mock(side_effect=[None, PermissionError("denied")]))
    with (tmp_path / "run.log").open("w") as log:
        with pytest.warns(RuntimeWarning, match="cleanup was denied"):
            assert run_logged(["unused"], log) == 0
    assert process.wait.call_count == 2


@pytest.mark.parametrize("stage", ["Ray startup", "Gym service startup"])
def test_gym_startup_deadline_reports_stage_and_stops_process(stage, tmp_path, monkeypatch):
    import subprocess

    import click

    from experiments import common, gym_runner

    marker = tmp_path / "terminated"
    driver = (
        "import signal, sys, time; from pathlib import Path; "
        f"signal.signal(signal.SIGTERM, lambda *_: (Path({str(marker)!r}).touch(), sys.exit(0))); "
    )
    if stage == "Gym service startup":
        driver += f"print({gym_runner.RAY_READY_MESSAGE!r}, flush=True); "
    driver += "time.sleep(60)"
    original = subprocess.Popen
    seen = []

    def launch(command, **kwargs):
        seen.append(command)
        return original([sys.executable, "-c", driver], **kwargs)

    monkeypatch.setattr(common.subprocess, "Popen", launch)
    monkeypatch.setattr(gym_runner, "RAY_STARTUP_TIMEOUT_S", 0.5)
    monkeypatch.setattr(gym_runner, "GYM_STARTUP_TIMEOUT_S", 0.5)
    with (tmp_path / "gym.log").open("w") as log:
        with pytest.raises(click.ClickException, match=stage + " exceeded"):
            run_logged(["gym", "eval", "run", "--config", "example.yaml"], log, timeout_s=10)
    assert marker.exists()
    assert seen[0][:3] == [sys.executable, "-m", "experiments.gym_runner"]


def test_gym_startup_readiness_relays_progress_and_preserves_exit_status(tmp_path, monkeypatch):
    import json
    import subprocess

    from experiments import common, gym_runner

    events = tmp_path / "progress.jsonl"
    monkeypatch.setenv("POC_WORKSHOP_EVENTS", str(events))
    driver = (
        "import time, sys; "
        f"print({gym_runner.RAY_READY_MESSAGE!r}, flush=True); "
        "print('All 2 / 2 servers ready! Polling every 60s', flush=True); "
        "time.sleep(0.3); sys.exit(7)"
    )
    original = subprocess.Popen
    monkeypatch.setattr(
        common.subprocess, "Popen", lambda command, **kw: original([sys.executable, "-c", driver], **kw)
    )
    with (tmp_path / "gym.log").open("w") as log:
        assert run_logged(["gym", "eval", "run"], log, timeout_s=5) == 7
    entry = json.loads(events.read_text())
    assert entry["task_id"] == "startup"
    assert "Gym services ready" in entry["message"]


def test_overall_deadline_still_bounds_gym_startup(tmp_path, monkeypatch):
    import subprocess

    from experiments import common, gym_runner

    original = subprocess.Popen
    monkeypatch.setattr(
        common.subprocess,
        "Popen",
        lambda command, **kw: original([sys.executable, "-c", "import time; time.sleep(60)"], **kw),
    )
    monkeypatch.setattr(gym_runner, "RAY_STARTUP_TIMEOUT_S", 10)
    with (tmp_path / "gym.log").open("w") as log:
        with pytest.raises(subprocess.TimeoutExpired) as caught:
            run_logged(["gym", "eval", "run"], log, timeout_s=0.3)
    assert caught.value.timeout == 0.3


def test_failed_service_startup_reports_stage_before_whole_run_deadline(tmp_path, monkeypatch):
    import subprocess

    import click

    from experiments import common, gym_runner

    original = subprocess.Popen
    driver = f"import sys; print({gym_runner.RAY_READY_MESSAGE!r}, flush=True); sys.exit(2)"
    monkeypatch.setattr(
        common.subprocess, "Popen", lambda command, **kw: original([sys.executable, "-c", driver], **kw)
    )
    with (tmp_path / "gym.log").open("w") as log:
        with pytest.raises(click.ClickException, match=r"Gym service startup failed \(Gym exit 2\)"):
            run_logged(["gym", "eval", "run"], log, timeout_s=300)
