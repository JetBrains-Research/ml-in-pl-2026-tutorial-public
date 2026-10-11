"""Stage the calculator and official Terminal-Bench task files at runtime."""

import json
import shlex
from pathlib import Path

from experiments.calculator import FIXTURE, setup_script
from experiments.ide_mcp.task import GOLD_SCRIPT, VERIFY_SCRIPT, Grade

MODULE_ROOT = Path(__file__).resolve().parent
PROJECT_PATH = "/tmp/ml-in-pl-calculator/project"
TASK_PATH = "/tmp/ml-in-pl-calculator/task"
CHECKS_MARKER = "__WORKSHOP_CHECKS__:"


def python_command(source: str) -> str:
    return shlex.join(["python3", "-I", "-c", source])


def reset_parameters() -> dict[str, str]:
    # The harness stages this directory before the official reset selects it.
    return {"task_id": "calculator-rename", "task_path": TASK_PATH}


def task_files(project_path: str = PROJECT_PATH) -> dict[str, str]:
    prompt = json.loads((MODULE_ROOT / "data/task.jsonl").read_text())
    verifier = VERIFY_SCRIPT.replace("/home/appuser/work", project_path)
    verifier = verifier.replace("print(json.dumps(results))", f"print({CHECKS_MARKER!r} + json.dumps(results))")
    verifier += "\nPath('/logs/verifier/reward.txt').write_text(str(float(all(results.values()))))\n"
    return {
        "instruction.md": prompt["responses_create_params"]["input"][-1]["content"] + "\n",
        "task.toml": "[verifier]\ntimeout_sec = 60\n",
        # Upstream reads WORKDIR as task metadata; it does not build this file in local mode.
        "environment/Dockerfile": f"WORKDIR {project_path}\n",
        "tests/verify.py": verifier,
        "tests/test.sh": "#!/bin/bash\nset -eu\npython3 -I /tests/verify.py\n",
    }


def runtime_setup(project_path: str = PROJECT_PATH, task_path: str = TASK_PATH) -> str:
    files = {name: content for name, content in task_files(project_path).items() if not name.startswith("tests/")}
    source = setup_script(project_path) + "; "
    source += f"task=Path({task_path!r}); shutil.rmtree(task, ignore_errors=True); task.mkdir(parents=True); "
    source += (
        f"files={files!r}; "
        "[(task / name).parent.mkdir(parents=True, exist_ok=True) for name in files]; "
        "[(task / name).write_text(content) for name, content in files.items()]"
    )
    return python_command(source)


def verifier_setup(project_path: str = PROJECT_PATH, task_path: str = TASK_PATH) -> str:
    # Send the trusted verifier only after the model finishes. This also works
    # with the official image's default TB2_WITHHOLD_TESTS=0.
    files = {name: content for name, content in task_files(project_path).items() if name.startswith("tests/")}
    source = (
        "from pathlib import Path; import shutil; "
        f"task=Path({task_path!r}); tests=task / 'tests'; "
        "tests.unlink() if tests.is_symlink() or tests.is_file() else shutil.rmtree(tests, ignore_errors=True); "
        f"files={files!r}; "
        "[(task / name).parent.mkdir(parents=True, exist_ok=True) for name in files]; "
        "[(task / name).write_text(content) for name, content in files.items()]"
    )
    return python_command(source)


def stage_task(context: Path) -> None:
    """Write the public task assets for local protocol tests; no image is built."""
    project = context / "project"
    project.mkdir(parents=True)
    for name, content in FIXTURE.items():
        (project / name).write_text(content)
    for name, content in task_files().items():
        path = context / "task" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)


def reference_command() -> str:
    return python_command(GOLD_SCRIPT.replace("/home/appuser/work", PROJECT_PATH))


def checks_from_output(output: str) -> Grade:
    for line in output.splitlines():
        if CHECKS_MARKER in line:
            return Grade.model_validate_json(line.split(CHECKS_MARKER, 1)[1].strip())
    raise RuntimeError("The official environment did not return all five verification results")
