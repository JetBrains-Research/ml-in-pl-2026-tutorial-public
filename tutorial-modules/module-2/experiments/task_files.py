"""Transfer bounded source files between task-agnostic agent and grading sandboxes."""

import json


def export_script(project_path: str, names: tuple[str, ...]) -> str:
    """Read only the expected small regular files; missing files remain failed submissions."""
    return (
        "import json; from pathlib import Path; "
        f"root=Path({project_path!r}); paths=[root/n for n in {names!r}]; "
        "assert all(not p.is_symlink() for p in paths), 'Submission contains a symlink'; "
        "assert all(not p.exists() or (p.is_file() and p.stat().st_size <= 65536) for p in paths), "
        "'Submission must contain small regular files'; "
        "print(json.dumps({p.name: p.read_text() if p.is_file() else None for p in paths}))"
    )


def parse_submission(output: str, names: tuple[str, ...]) -> dict[str, str | None]:
    submission = json.loads(output)
    if (
        not isinstance(submission, dict)
        or set(submission) != set(names)
        or not all(value is None or isinstance(value, str) for value in submission.values())
    ):
        raise RuntimeError("Invalid submission file manifest")
    return submission


def restore_script(project_path: str, submission: dict[str, str | None]) -> str:
    # A fresh grader keeps its own dependencies and IDE settings. Clear its task
    # sources so a missing submission cannot silently use an original fixture file.
    files = {name: source for name, source in submission.items() if source is not None}
    return (
        "from pathlib import Path; import shutil; "
        f"root=Path({project_path!r}); "
        "[shutil.rmtree(p) if p.is_dir() else p.unlink() for p in root.iterdir() if p.name != '.idea']; "
        f"[(root / name).write_text(source) for name, source in {files!r}.items()]"
    )
