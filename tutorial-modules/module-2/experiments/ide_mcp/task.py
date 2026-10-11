"""Cross-file refactoring and independent checks for bundled PyCharm MCP."""

import json
import shlex
from typing import Literal

from pydantic import BaseModel

TASK_ID = "calculator-rename"
PROJECT_PATH = "/home/appuser/work"
PYTHON = "/opt/idegym/.venv/bin/python"
Mode = Literal["bundled"]

# These files come from the pinned IdeGYM project's Python test fixture. The gold
# transformation is used only by the verifier smoke run, never sent to the model.
GOLD_SCRIPT = r"""
from pathlib import Path
import re
for name in ('calculator.py', 'main.py', 'test_calculator.py'):
    path = Path('/home/appuser/work') / name
    path.write_text(re.sub(r'\badd\b', 'sum_numbers', path.read_text()))
"""

VERIFY_SCRIPT = """
import ast
import contextlib
import importlib
import io
import json
import sys
from pathlib import Path

root = Path('/home/appuser/work')
sys.path.insert(0, str(root))
results = {}

def check(name, operation):
    try:
        results[name] = bool(operation())
    except Exception:
        results[name] = False

def renamed():
    module = importlib.import_module('calculator')
    return callable(getattr(module, 'sum_numbers', None)) and not hasattr(module, 'add')

def references_updated():
    for name in ('main.py', 'test_calculator.py'):
        tree = ast.parse((root / name).read_text())
        imports = [alias.name for node in ast.walk(tree)
                   if isinstance(node, ast.ImportFrom) and node.module == 'calculator'
                   for alias in node.names]
        calls = [node.func.id for node in ast.walk(tree)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
        if 'sum_numbers' not in imports or 'sum_numbers' not in calls or 'add' in imports or 'add' in calls:
            return False
    return True

def arithmetic():
    c = importlib.import_module('calculator')
    return (c.sum_numbers(2, 3) == 5 and c.sum_numbers(-4, 1) == -3
            and c.sum_numbers(0, 0) == 0 and c.sum_numbers(1.5, 2.25) == 3.75
            and c.subtract(10, 4) == 6 and c.multiply(6, 7) == 42 and c.divide(15, 3) == 5)

def zero_division():
    try:
        importlib.import_module('calculator').divide(1, 0)
    except ValueError as exc:
        return str(exc) == 'Cannot divide by zero'
    return False

def main_output():
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        importlib.import_module('main').main()
    return output.getvalue() == '2 + 3 = 5\\n10 - 4 = 6\\n6 * 7 = 42\\n15 / 3 = 5.0\\n'

for name, operation in [('renamed', renamed), ('references_updated', references_updated),
                        ('arithmetic', arithmetic), ('zero_division', zero_division), ('main_output', main_output)]:
    check(name, operation)
print(json.dumps(results))
"""


def python_command(source: str) -> str:
    return shlex.join([PYTHON, "-I", "-c", source])


class Grade(BaseModel):
    renamed: bool
    references_updated: bool
    arithmetic: bool
    zero_division: bool
    main_output: bool

    @property
    def reward(self) -> float:
        return float(all(self.model_dump().values()))


def parse_grade(stdout: str, exit_code: int) -> Grade:
    if exit_code != 0:
        raise RuntimeError("IDE verifier failed to execute; this is not a model score")
    return Grade.model_validate(json.loads(stdout))
