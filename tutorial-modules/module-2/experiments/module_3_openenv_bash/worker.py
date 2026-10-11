"""Run the official OpenEnv interface in a worker using the project Python environment."""

import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.module_3_openenv_bash.workshop import run_client


def emit(data: dict) -> None:
    print(json.dumps({"ok": True, "data": data}), flush=True)


if __name__ == "__main__":
    try:
        initial = json.loads(sys.stdin.readline())
        asyncio.run(run_client(initial["base_url"], initial["reset"], sys.stdin.readline, emit))
    except Exception as error:
        message = str(error)
        for key, value in os.environ.items():
            if value and any(part in key for part in ("PASSWORD", "USERNAME", "API_KEY")):
                message = message.replace(value, "[REDACTED]")
        print(f"OpenEnv client failed ({type(error).__name__}): {message}", file=sys.stderr)
        print(json.dumps({"ok": False}), flush=True)
        sys.exit(1)
