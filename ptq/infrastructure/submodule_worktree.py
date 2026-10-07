"""Create a git worktree whose submodules are cloned from the local checkout.

Same result as PyTorch's ``tools/create_worktree.py create`` (each submodule
is a local clone of the source checkout's submodule, pinned to the commit the
parent expects, with origin pointed back at the real URL), but faster:

- clones use ``--no-checkout`` and then check out the pinned commit once,
  instead of checking out the default branch and then switching;
- submodules are cloned in parallel; nested submodules are queued as soon as
  their parent is checked out;
- large checkouts (cutlass, composable_kernel, ...) use git's parallel
  checkout, since they are the critical path once everything else is done.

On the seed this takes ~3s versus ~25s for the upstream script.
"""

from __future__ import annotations

import configparser
import subprocess
import sys
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path

MAX_WORKERS = 16
CHECKOUT_WORKERS = 4


@dataclass(frozen=True)
class _Submodule:
    source_root: Path  # repo containing the submodule in the source checkout
    worktree_root: Path  # corresponding repo in the new worktree
    path: str
    url: str


def _git(*args: str, cwd: Path | None = None) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout


def _parse_gitmodules(root: Path) -> list[tuple[str, str]]:
    gitmodules = root / ".gitmodules"
    if not gitmodules.exists():
        return []
    config = configparser.ConfigParser()
    config.read(gitmodules)
    return [
        (config.get(s, "path"), config.get(s, "url"))
        for s in config.sections()
        if s.startswith('submodule "')
    ]


def _resolve_git_dir(checkout: Path) -> Path | None:
    git_path = checkout / ".git"
    if git_path.is_dir():
        return git_path
    if git_path.is_file():
        content = git_path.read_text().strip()
        if content.startswith("gitdir: "):
            resolved = (checkout / content[len("gitdir: ") :]).resolve()
            if resolved.exists():
                return resolved
    return None


def _children(source_root: Path, worktree_root: Path) -> list[_Submodule]:
    return [
        _Submodule(source_root, worktree_root, path, url)
        for path, url in _parse_gitmodules(worktree_root)
    ]


def _clone(sub: _Submodule) -> list[_Submodule]:
    source = sub.source_root / sub.path
    dest = sub.worktree_root / sub.path
    git_dir = _resolve_git_dir(source)
    if git_dir is None:
        return []

    out = _git("ls-tree", "HEAD", sub.path, cwd=sub.source_root).split()
    commit = out[2] if len(out) >= 3 else None

    # The parent checkout leaves an empty directory at the submodule path.
    if dest.exists():
        try:
            dest.rmdir()
        except OSError:
            pass

    _git("clone", "--quiet", "--no-checkout", str(git_dir), str(dest))
    _git(
        "-c",
        f"checkout.workers={CHECKOUT_WORKERS}",
        "checkout",
        "--quiet",
        commit or "HEAD",
        cwd=dest,
    )
    _git("remote", "set-url", "origin", sub.url, cwd=dest)
    # Register nested submodules so `git submodule status --recursive` sees them.
    subprocess.run(["git", "submodule", "init"], cwd=dest, capture_output=True)
    return _children(source, dest)


def create_worktree(
    source_repo: str | Path,
    worktree_path: str | Path,
    commit: str = "HEAD",
    *,
    max_workers: int = MAX_WORKERS,
) -> list[str]:
    """Create the worktree; return warnings for submodules that failed to clone."""
    source_repo = Path(source_repo).expanduser()
    worktree_path = Path(worktree_path).expanduser()
    _git("worktree", "add", "--quiet", str(worktree_path), commit, cwd=source_repo)

    warnings: list[str] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        pending: dict[Future[list[_Submodule]], _Submodule] = {
            pool.submit(_clone, s): s for s in _children(source_repo, worktree_path)
        }
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for fut in done:
                sub = pending.pop(fut)
                try:
                    nested = fut.result()
                except subprocess.CalledProcessError as e:
                    rel = (sub.worktree_root / sub.path).relative_to(worktree_path)
                    warnings.append(f"failed to clone {rel}: {(e.stderr or '').strip()}")
                    continue
                for child in nested:
                    pending[pool.submit(_clone, child)] = child
    return warnings


if __name__ == "__main__":
    for w in create_worktree(sys.argv[1], sys.argv[2], *sys.argv[3:4]):
        print(f"WARNING: {w}", file=sys.stderr)
