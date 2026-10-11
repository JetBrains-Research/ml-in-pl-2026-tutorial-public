"""Materialize inspection scores from saved verification evidence."""

import json
from pathlib import Path

from experiments.module_4_jetbrains_mcp.evaluation import InspectionResult


def write_summary(output: Path, settings: dict) -> InspectionResult:
    result = InspectionResult.model_validate_json((output / "verification.json").read_text())
    (output / "summary.json").write_text(json.dumps(settings | result.model_dump(), indent=2) + "\n")
    return result
