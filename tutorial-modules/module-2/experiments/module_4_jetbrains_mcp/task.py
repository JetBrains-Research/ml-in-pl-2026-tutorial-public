"""Seeded inspection task and independent behavioral verification."""

import json
from pathlib import Path

from experiments.ide_mcp.task import PROJECT_PATH, python_command
from experiments.ide_mcp.transport import IdeConnection, ToolResult

TASK_ID = "ide-inspections"
FILES = ("pricing.py", "receipt.py")
BROKEN = {
    "pricing.py": (
        "def discounted_total(price: float, quantity: int, discount: float) -> float:\n"
        "    return price * quantity * (1 - discount_rate)\n"
    ),
    "receipt.py": (
        "from pricing import discounted_total\n\n"
        "def format_receipt(price: float, quantity: int, discount: float) -> str\n"
        "    total = discounted_total(price, quantity, discount)\n"
        "    return f'Total: {total:.2f}'\n"
    ),
}
FIXED = {
    "pricing.py": BROKEN["pricing.py"].replace("discount_rate", "discount"),
    "receipt.py": BROKEN["receipt.py"].replace("-> str\n", "-> str:\n"),
}
VERIFY_SCRIPT = r"""
import ast
import importlib
import json
import sys
from pathlib import Path
root = Path('/home/appuser/work')
sys.path.insert(0, str(root))
checks = dict(files_present=False, no_suppressions=False, pricing=False, receipt=False)
try:
    sources = [(root / name).read_text() for name in ('pricing.py', 'receipt.py')]
    checks['files_present'] = all(s.strip() for s in sources)
    checks['no_suppressions'] = all(not any(marker in s.lower() for marker in
        ('noinspection', 'type: ignore', 'noqa', '# pyright:', 'inspectionprofile')) for s in sources)
    for source in sources:
        ast.parse(source)
    pricing = importlib.import_module('pricing')
    receipt = importlib.import_module('receipt')
    checks['pricing'] = all(abs(pricing.discounted_total(price=p, quantity=q, discount=d) - expected) < 1e-9
        for p, q, d, expected in ((10.0, 2, 0.1, 18.0), (2.5, 3, 0.2, 6.0),
            (9.0, 0, 0.5, 0.0), (4.0, 2, 0.0, 8.0), (5.0, 3, 1.0, 0.0)))
    checks['receipt'] = all(receipt.format_receipt(price=p, quantity=q, discount=d) == expected
        for p, q, d, expected in ((10.0, 2, 0.1, 'Total: 18.00'),
            (2.5, 3, 0.2, 'Total: 6.00'), (4.0, 2, 0.0, 'Total: 8.00')))
except Exception:
    pass
print(json.dumps(checks))
"""


def write_files_script(files: dict[str, str], *, clean: bool = False) -> str:
    # Only the harness seeds the fixture; reference fixes are never sent to the model.
    return (
        "from pathlib import Path; import shutil; "
        f"root=Path({PROJECT_PATH!r}); "
        + (
            "[shutil.rmtree(p) if p.is_dir() else p.unlink() for p in root.iterdir() if p.name != '.idea']; "
            if clean
            else ""
        )
        + f"[(root / name).write_text(source) for name, source in {files!r}.items()]"
    )


def parse_problems(result: ToolResult) -> list[dict]:
    if result.is_error:
        raise RuntimeError("PyCharm inspection call failed: " + result.model_dump_json())
    texts = [block["text"] for block in result.content if block.get("type") == "text"]
    payload = json.loads("\n".join(texts))
    if isinstance(payload, dict) and isinstance(payload.get("errors"), list):
        problems = payload["errors"]
    elif isinstance(payload, dict) and isinstance(payload.get("problems"), list):
        problems = payload["problems"]
    elif isinstance(payload, list):
        problems = payload
    else:
        raise RuntimeError("Unexpected PyCharm inspection response: " + repr(payload))
    if not all(isinstance(problem, dict) for problem in problems):
        raise RuntimeError("Invalid inspection diagnostics")
    return problems


async def inspect(
    connection: IdeConnection, output: Path, phase: str, *, files: tuple[str, ...] = FILES
) -> dict[str, list[dict]]:
    results = {}
    for name in files:
        result = await connection.call(
            "pycharm_get_file_problems",
            {
                "filePath": name,
                "projectPath": PROJECT_PATH,
                "errorsOnly": False,
                "timeout": 60000,
            },
        )
        # Save the raw response even when parsing or inspection execution fails.
        (output / f"{phase}-{name}.json").write_text(result.model_dump_json(indent=2) + "\n")
        results[name] = parse_problems(result)
    (output / f"inspections-{phase}.json").write_text(json.dumps(results, indent=2) + "\n")
    return results


async def seed(connection: IdeConnection) -> None:
    result = await connection.server.execute_bash(
        script=python_command(write_files_script(BROKEN, clean=True)),
        command_timeout=30,
    )
    if result.exit_code:
        raise RuntimeError("Could not seed the inspection fixture")
