"""Only the expected source files cross into a clean grading sandbox."""

import json
import subprocess
import sys

import pytest

from experiments.task_files import export_script, parse_submission, restore_script


def test_source_transfer_keeps_missing_files_missing_and_preserves_grader_settings(tmp_path):
    agent, grader = tmp_path / "agent", tmp_path / "grader"
    agent.mkdir()
    grader.mkdir()
    names = ("one.py", "missing.py")
    (agent / "one.py").write_text("print('submitted')")
    (agent / "unrelated.txt").write_text("must stay in agent sandbox")
    (grader / "missing.py").write_text("original fixture")
    settings = grader / ".idea"
    settings.mkdir()
    (settings / "profile.xml").write_text("fresh grading settings")
    exported = subprocess.run(
        [sys.executable, "-I", "-c", export_script(str(agent), names)], capture_output=True, text=True, check=True
    )
    submission = parse_submission(exported.stdout, names)
    assert submission == {"one.py": "print('submitted')", "missing.py": None}
    subprocess.run([sys.executable, "-I", "-c", restore_script(str(grader), submission)], check=True)
    assert (grader / "one.py").read_text() == submission["one.py"]
    assert not (grader / "missing.py").exists()
    assert not (grader / "unrelated.txt").exists()
    assert (settings / "profile.xml").read_text() == "fresh grading settings"


@pytest.mark.parametrize("invalid", ["symlink", "directory", "oversized"])
def test_export_rejects_files_outside_submission_bounds(tmp_path, invalid):
    source = tmp_path / "one.py"
    if invalid == "symlink":
        target = tmp_path / "target.txt"
        target.write_text("not an allowed source")
        source.symlink_to(target)
    elif invalid == "directory":
        source.mkdir()
    else:
        source.write_text("x" * 65537)
    result = subprocess.run(
        [sys.executable, "-I", "-c", export_script(str(tmp_path), ("one.py",))], capture_output=True, text=True
    )
    assert result.returncode != 0


@pytest.mark.parametrize("manifest", [[], {}, {"one.py": 123}, {"one.py": "source", "extra.py": "source"}])
def test_invalid_submission_manifest_is_an_error(manifest):
    with pytest.raises(RuntimeError, match="manifest"):
        parse_submission(json.dumps(manifest), ("one.py",))
