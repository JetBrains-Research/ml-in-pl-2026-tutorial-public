"""Shared calculator files; runners stage them at runtime instead of baking a task image."""

import json
from pathlib import Path

FIXTURE = json.loads((Path(__file__).resolve().parents[1] / "data/calculator.json").read_text())


def setup_script(project_path: str) -> str:
    """Write a clean calculator project without touching server packages or configuration."""
    return (
        "from pathlib import Path; import shutil; "
        f"root=Path({project_path!r}); root.mkdir(parents=True, exist_ok=True); "
        "[shutil.rmtree(p) if p.is_dir() else p.unlink() for p in root.iterdir() if p.name != '.idea']; "
        f"[(root / name).write_text(source) for name, source in {FIXTURE!r}.items()]"
    )
