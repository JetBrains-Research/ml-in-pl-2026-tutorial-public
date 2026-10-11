"""Inspection schemas and tool scoping; workshop.py owns the two-sandbox episode."""

import json
from pathlib import Path, PurePosixPath

from pydantic import BaseModel

from experiments.ide_mcp.task import PROJECT_PATH
from experiments.ide_mcp.transport import IdeConnection, ToolResult
from experiments.module_4_jetbrains_mcp.task import (
    FILES,
)

MODEL_TOOLS = {
    "pycharm_get_file_problems",
    "pycharm_get_file_text_by_path",
    "pycharm_replace_text_in_file",
    "pycharm_reformat_file",
}
PATH_FIELDS = {
    "pycharm_get_file_problems": "filePath",
    "pycharm_get_file_text_by_path": "pathInProject",
    "pycharm_replace_text_in_file": "pathInProject",
    "pycharm_reformat_file": "path",
}
EDIT_TOOLS = {"pycharm_replace_text_in_file", "pycharm_reformat_file"}


class ScopedInspectionConnection(IdeConnection):
    """Keep model calls within the two exercise files; grader calls use the original connection."""

    async def call(self, name: str, arguments: dict) -> ToolResult:
        path = arguments.get(PATH_FIELDS.get(name, ""), "")
        path = str(PurePosixPath(path)) if isinstance(path, str) and path else ""
        if name not in MODEL_TOOLS or path not in (*FILES, *(f"{PROJECT_PATH}/{n}" for n in FILES)):
            return ToolResult(
                is_error=True, content=[{"type": "text", "text": "Use a task file: pricing.py or receipt.py"}]
            )
        if arguments.get("projectPath", PROJECT_PATH) != PROJECT_PATH:
            return ToolResult(is_error=True, content=[{"type": "text", "text": "Use projectPath=/home/appuser/work"}])
        arguments = arguments | {"projectPath": PROJECT_PATH}
        if name == "pycharm_get_file_problems":
            arguments["errorsOnly"] = False
        return await super().call(name, arguments)


def inspected_before_and_after_edits(entries: list[dict]) -> bool:
    successful = [entry for entry in entries if entry.get("tool") and not entry.get("is_error", True)]
    edits = [i for i, entry in enumerate(successful) if entry["tool"] in EDIT_TOOLS]
    if not edits:
        return False

    def inspected(rows: list[dict]) -> set[str]:
        return {
            Path(json.loads(entry["arguments"]).get("filePath", "")).name
            for entry in rows
            if entry["tool"] == "pycharm_get_file_problems"
        }

    return set(FILES) <= inspected(successful[: edits[0]]) and set(FILES) <= inspected(successful[edits[-1] + 1 :])


class BehaviorChecks(BaseModel):
    files_present: bool
    no_suppressions: bool
    pricing: bool
    receipt: bool


class InspectionChecks(BaseModel):
    baseline_detected: bool
    inspections_clean: bool
    behavior_preserved: bool
    model_used_inspections: bool

    @property
    def reward(self) -> float:
        return float(all(self.model_dump().values()))


class InspectionResult(BaseModel):
    reward: float
    checks: InspectionChecks
    golden: bool
    before_counts: dict[str, int]
    after_counts: dict[str, int]
    behavior: dict[str, bool]
    tool_calls: int
    successful_tool_calls: int


async def evaluate(*args, **kwargs):
    """Compatibility entry point; task orchestration is taught in workshop.py."""
    from experiments.module_4_jetbrains_mcp.workshop import evaluate as workshop_evaluate

    return await workshop_evaluate(*args, **kwargs)
