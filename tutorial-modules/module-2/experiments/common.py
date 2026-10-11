# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared configuration and subprocess lifecycle for the evaluation modules."""

import json
import os
import re
import signal
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

import click
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]


def progress(task_id: str, message: str) -> None:
    path = os.environ.get("POC_WORKSHOP_EVENTS")
    if path:
        event = {"timestamp": datetime.now(timezone.utc).isoformat(), "task_id": task_id, "message": message}
        with Path(path).open("a") as stream:
            stream.write(json.dumps(event) + "\n")


def model_for_provider(model: str) -> str:
    """Translate known model aliases without changing the selected model."""
    gemma_names = {
        "litellm": "vertex_ai/google/gemma-4-26b-a4b-it-maas",
        "openrouter": "google/gemma-4-26b-a4b-it",
    }
    if model in gemma_names.values():
        provider = os.environ.get("MODEL_PROVIDER", "litellm").strip().lower()
        return gemma_names[provider]
    return model


def configure(module_root: Path) -> None:
    """Load credentials into the process, without putting them on command lines."""
    load_dotenv(ROOT / ".env", override=False)
    idegym_prefix = os.environ.get("IDEGYM_ENV_PREFIX", "IDEGYM_WORKSHOP").strip()
    if not idegym_prefix:
        raise click.ClickException("IDEGYM_ENV_PREFIX must not be empty")
    idegym_settings = {
        f"IDEGYM_{suffix}": f"{idegym_prefix}_{suffix}"
        for suffix in ("ORCHESTRATOR_URL", "AUTH_USERNAME", "AUTH_PASSWORD")
    }
    provider = os.environ.get("MODEL_PROVIDER", "litellm").strip().lower()
    if provider not in ("litellm", "openrouter"):
        raise click.ClickException("MODEL_PROVIDER must be litellm or openrouter")
    prefix = provider.upper()
    required = (
        *idegym_settings.values(),
        f"{prefix}_BASE_URL",
        f"{prefix}_API_KEY",
    )
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise click.ClickException("Missing settings in .env: " + ", ".join(missing))
    # Gym configs and client subprocesses consume the canonical variable names.
    for target, source in idegym_settings.items():
        os.environ[target] = os.environ[source]
    os.environ["IDEGYM_NAMESPACE"] = os.environ.get(f"{idegym_prefix}_NAMESPACE", "idegym")
    base = os.environ[f"{prefix}_BASE_URL"].rstrip("/")
    os.environ["POC_MODEL_BASE_URL"] = base if base.endswith("/v1") else base + "/v1"
    os.environ["POC_MODEL_API_KEY"] = os.environ[f"{prefix}_API_KEY"]
    default_model = "vertex_ai/google/gemma-4-26b-a4b-it-maas"
    os.environ["POC_MODEL_NAME"] = model_for_provider(
        os.environ.get(f"{prefix}_MODEL_NAME_{module_root.name.upper()}")
        or os.environ.get(f"POLICY_MODEL_NAME_{module_root.name.upper()}")
        or os.environ.get(f"{prefix}_MODEL_NAME")
        or os.environ.get("POLICY_MODEL_NAME")
        or default_model
    )
    os.environ.setdefault("RAY_TMPDIR", "/tmp")
    os.environ.setdefault("RAY_USAGE_STATS_ENABLED", "0")
    # Gym starts components in separate directories; all use our existing venv.
    os.environ["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "0"
    from experiments.artifacts import current_output

    output = current_output()
    os.environ.setdefault("POC_RUNTIME_DIR", str(output / "runtime" if output else ROOT / ".runtime"))
    os.environ.setdefault("POC_LOG_DIR", str(ROOT / ".runtime/logs"))
    extra_roots = os.environ.get("NEMO_GYM_EXTRA_ROOTS", "").split(os.pathsep)
    os.environ["NEMO_GYM_EXTRA_ROOTS"] = os.pathsep.join(dict.fromkeys([str(module_root), *filter(None, extra_roots)]))
    # IDE components import shared experiment code after Gym changes their cwd.
    python_paths = os.environ.get("PYTHONPATH", "").split(os.pathsep)
    os.environ["PYTHONPATH"] = os.pathsep.join(dict.fromkeys([str(ROOT), *filter(None, python_paths)]))
    os.chdir(ROOT)
    os.umask(0o077)


def prepare_runtime(components: tuple[str, ...]) -> None:
    """Reuse the locked project venv for Gym component subprocesses."""
    runtime = Path(os.environ.get("POC_RUNTIME_DIR", str(ROOT / ".runtime")))
    for component in components:
        directory = runtime / component
        directory.mkdir(parents=True, exist_ok=True)
        link = directory / ".venv"
        if not link.exists():
            link.symlink_to(Path(sys.prefix), target_is_directory=True)


def wait_for_gym_startup(process: subprocess.Popen, log: TextIO, *, deadline: float | None) -> None:
    from experiments.gym_runner import GYM_STARTUP_TIMEOUT_S, RAY_READY_MESSAGE, RAY_STARTUP_TIMEOUT_S

    stage = "Ray startup"
    stage_timeout = RAY_STARTUP_TIMEOUT_S
    stage_deadline = time.monotonic() + stage_timeout
    with Path(log.name).open() as stream:
        while True:
            for line in iter(stream.readline, ""):
                if RAY_READY_MESSAGE in line and stage == "Ray startup":
                    stage = "Gym service startup"
                    stage_timeout = GYM_STARTUP_TIMEOUT_S
                    stage_deadline = time.monotonic() + stage_timeout
                if re.search(r"All (\d+) / \1 servers ready!", line):
                    progress("startup", "Gym services ready. Starting the model and remote sandbox episode.")
                    return
            code = process.poll()
            if code is not None:
                if code:
                    raise click.ClickException(
                        f"{stage} failed (Gym exit {code}); local Gym processes stopped. "
                        f"Incomplete evaluation, not reward 0. Inspect {log.name}."
                    )
                return
            now = time.monotonic()
            if deadline is not None and now >= deadline:
                raise subprocess.TimeoutExpired(process.args, 0)
            if now >= stage_deadline:
                raise click.ClickException(
                    f"{stage} exceeded {stage_timeout}s; local Gym processes stopped. "
                    f"Incomplete evaluation, not reward 0. Inspect {log.name}."
                )
            time.sleep(0.1)


def run_logged(command: list[str], log: TextIO, *, timeout_s: float | None = None) -> int:
    """Contain Gym's shell pipelines in one process group and reap them on exit."""
    output = os.environ.get("POC_RUN_DIR")
    gym_run = command[1:3] == ["eval", "run"]
    if output and gym_run:
        command = [*command, f"hydra.run.dir={output}/hydra", "hydra.job.chdir=false"]
    if gym_run:
        command = [sys.executable, "-m", "experiments.gym_runner", *command[1:]]
    started = time.monotonic()
    deadline = started + timeout_s if timeout_s is not None else None
    process = subprocess.Popen(
        command,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    try:
        if gym_run:
            try:
                wait_for_gym_startup(process, log, deadline=deadline)
            except subprocess.TimeoutExpired as error:
                raise subprocess.TimeoutExpired(command, timeout_s) from error
        remaining = max(0, deadline - time.monotonic()) if deadline is not None else None
        try:
            return process.wait(timeout=remaining)
        except subprocess.TimeoutExpired as error:
            raise subprocess.TimeoutExpired(command, timeout_s) from error
    finally:
        # This branch kills shell wrappers after a short grace period, which can leave
        # their Python children alive. Signal only this invocation's process group.
        try:
            os.killpg(process.pid, signal.SIGTERM)
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                process.poll()
                os.killpg(process.pid, 0)
                time.sleep(0.1)
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except PermissionError:
            if process.poll() is None:
                raise
            # macOS can deny signals/probes to a group after its driver has exited.
            # Preserve the exit status, but make incomplete descendant cleanup visible.
            warnings.warn("Process-group cleanup was denied after the driver exited", RuntimeWarning, stacklevel=2)
        process.wait()
