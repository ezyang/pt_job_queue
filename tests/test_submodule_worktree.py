from __future__ import annotations

import subprocess
from pathlib import Path

from ptq.infrastructure.submodule_worktree import create_worktree


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "protocol.file.allow=always", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _repo(path: Path, filename: str) -> Path:
    path.mkdir()
    _git(path, "init", "-q", "-b", "main")
    _git(path, "config", "user.email", "t@example.com")
    _git(path, "config", "user.name", "t")
    (path / filename).write_text(filename)
    _git(path, "add", ".")
    _git(path, "commit", "-qm", filename)
    return path


def test_clones_nested_submodules_at_pinned_commits(tmp_path: Path):
    leaf = _repo(tmp_path / "leaf", "leaf.txt")
    mid = _repo(tmp_path / "mid", "mid.txt")
    _git(mid, "submodule", "add", "-q", str(leaf), "deps/leaf")
    _git(mid, "commit", "-qm", "add leaf")
    src = _repo(tmp_path / "src", "src.txt")
    _git(src, "submodule", "add", "-q", str(mid), "third_party/mid")
    _git(src, "commit", "-qm", "add mid")
    _git(src, "submodule", "update", "--init", "--recursive", "-q")

    # Move the upstream leaf past the pinned commit; the worktree must keep the pin.
    pinned_leaf = _git(leaf, "rev-parse", "HEAD")
    (leaf / "leaf.txt").write_text("moved")
    _git(leaf, "commit", "-qam", "move")

    dest = tmp_path / "jobs" / "job" / "src"
    assert create_worktree(src, dest) == []

    assert (dest / "src.txt").exists()
    assert (dest / "third_party/mid/mid.txt").exists()
    assert (dest / "third_party/mid/deps/leaf/leaf.txt").read_text() == "leaf.txt"
    assert _git(dest / "third_party/mid/deps/leaf", "rev-parse", "HEAD") == pinned_leaf
    assert _git(dest / "third_party/mid", "remote", "get-url", "origin") == str(mid)
    status = _git(dest, "submodule", "status", "--recursive")
    assert len(status.splitlines()) == 2
    assert not any(line.startswith(("-", "+")) for line in status.splitlines())


def test_skips_uninitialized_submodules(tmp_path: Path):
    dep = _repo(tmp_path / "dep", "dep.txt")
    src = _repo(tmp_path / "src", "src.txt")
    _git(src, "submodule", "add", "-q", str(dep), "dep")
    _git(src, "commit", "-qm", "add dep")
    _git(src, "submodule", "deinit", "-q", "--all")

    dest = tmp_path / "wt"
    assert create_worktree(src, dest) == []
    assert not (dest / "dep" / "dep.txt").exists()
