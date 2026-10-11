import asyncio
import json
import logging
import os
import shlex
from pathlib import Path

import structlog
from dotenv import load_dotenv
from idegym.client.client import IdeGYMClient, ServerCloseAction

load_dotenv()
structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.INFO))

IMAGE = "ghcr.io/jetbrains-research/idegym/server-debian-bookworm-20250520-slim:0.13.0"
PYTHON = "/opt/idegym/.venv/bin/python"
RUNNER = (
    "import json, runpy, sys; "
    "solve = runpy.run_path('/tmp/solution.py')['solve']; "
    "print(json.dumps(solve(json.loads(sys.argv[1]))))"
)

# Exercise: break one of these, e.g. use str.isalnum() in palindrome, and rerun.
SOLUTIONS = {
    "sum-list": "def solve(data):\n    return sum(data)\n",
    "count-vowels": "def solve(data):\n    return sum(c in 'aeiou' for c in data.lower())\n",
    "sorted-unique": "def solve(data):\n    return sorted(set(data))\n",
    "factorial": "import math\n\ndef solve(data):\n    return math.factorial(data)\n",
    "palindrome": (
        "import string\n\n"
        "def solve(data):\n"
        "    kept = [c.lower() for c in data if c in string.ascii_letters + string.digits]\n"
        "    return kept == kept[::-1]\n"
    ),
}


async def main() -> None:
    rows = [json.loads(line) for line in Path("data/tiny_python.jsonl").read_text().splitlines()]
    async with IdeGYMClient(
        orchestrator_url=os.environ["IDEGYM_ORCHESTRATOR_URL"],
        name=os.environ.get("IDEGYM_CLIENT_NAME", "ml-in-pl"),
        namespace=os.environ.get("IDEGYM_NAMESPACE", "idegym"),
    ) as client:
        async with client.with_server(
            image_tag=IMAGE,
            server_name="dataset",
            close_action=ServerCloseAction.STOP,
        ) as server:
            for row in rows:
                task = row["verifier_metadata"]
                await server.create_file(file_path="/tmp/solution.py", content=SOLUTIONS[task["task_id"]])
                passed = 0
                for case in task["test_cases"]:
                    command = shlex.join([PYTHON, "-I", "-c", RUNNER, json.dumps(case["input"])])
                    result = await server.execute_bash(script=command, command_timeout=5)
                    passed += result.exit_code == 0 and json.loads(result.stdout) == case["expected"]
                print(f"{task['task_id']:<15} {passed}/{len(task['test_cases'])}")


if __name__ == "__main__":
    asyncio.run(main())
