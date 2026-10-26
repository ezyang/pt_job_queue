"""reseed.py: give a job the seed's compiled torch instead of rebuilding.

    python3 -I $WORKSPACE/scripts/reseed.py WORKTREE [--force]
    python3 -I $WORKSPACE/scripts/reseed.py WORKTREE --auto   (rebuild.sh)

A job's compiled torch comes from the seed it was provisioned from (the
fast path in venv_service.py). The seed is rebuilt every night at a newer
commit, and jobs rebase onto it (the OSS DMZ hands it out as the base). A
job that moves onto it has source newer than its compiled torch, and daily
seed moves nearly always touch core headers or codegen, so a rebuild in
the job is close to a full one (15-60 minutes). Copying the seed's build
products takes seconds.

PyTorch's editable install is scikit-build-core: the venv's finder
(``_editable_skbc_torch.py``) loads Python from the worktree and the
compiled torch from ``build/install/<tag>/``. So this copies, from the
seed into the job: ``build/install``, the gitignored generated files in
the source tree (``.pyi`` stubs, generated sources), and the finder and
dist-info, with the seed's paths rewritten to the job's.

``--auto`` is for rebuild.sh: copy only if the job's source contains a
newer seed build and nothing that feeds the compiled build changed since
(committed or not); exit 0 if the job's torch then matches its source (no
build needed), 3 if it needs a build.

An OSS DMZ agent runs it in its sandbox, where the seed is read-only and
the job dir writable: it does nothing the agent couldn't do by hand. A seed
rebuild that starts mid-copy deletes files under it (``git clean -ffdx``);
the copy then fails, or the final check that torch imports at the seed's
commit does.

stdlib-only, Python 3.9+ (runs with the system python3 too).
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

FINDER = ("_editable_skbc_torch.py", "_editable_skbc_torch.pth")
# Changes under these (or with these suffixes) feed the compiled build.
COMPILED_DIRS = ("aten/", "c10/", "caffe2/", "cmake/", "third_party/", "torch/csrc/", "torchgen/",
                 "tools/autograd/", "tools/setup_helpers/", "functorch/csrc/", "torch/headeronly/")
COMPILED_SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".cu", ".cuh", ".yaml", ".cmake", ".in",
                     ".S", ".metal", ".mm", ".m", ".ld"}
COMPILED_NAMES = {"CMakeLists.txt", "setup.py", "pyproject.toml"}


class ReseedError(Exception):
    pass


def built_commit(src: Path) -> str | None:
    """The commit a checkout's compiled torch was built from (its generated version.py)."""
    for v in [*sorted(src.glob("build/install/*/torch/version.py")), src / "torch" / "version.py"]:
        try:
            m = re.search(r"^git_version\s*=\s*'([0-9a-f]{40})'", v.read_text(), re.M)
        except OSError:
            continue
        if m:
            return m.group(1)
    return None


def compiled_changes(names: list[str]) -> list[str]:
    def compiled(n: str) -> bool:
        if n.startswith("test/"):
            return False
        return (n.startswith(COMPILED_DIRS) or Path(n).suffix in COMPILED_SUFFIXES
                or Path(n).name in COMPILED_NAMES)
    return [n for n in names if compiled(n)]


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True,
                          stdin=subprocess.DEVNULL, timeout=300, check=check)


def changed_since(wt: Path, sha: str) -> list[str]:
    """Files feeding the compiled build that differ from ``sha`` in the
    worktree: committed, staged, unstaged or untracked."""
    names = _git(wt, "diff", "--name-only", sha).stdout.split()
    names += _git(wt, "ls-files", "--others", "--exclude-standard").stdout.split()
    return compiled_changes(names)


def _site_packages(venv: Path) -> Path:
    found = sorted(venv.glob("lib/python3*/site-packages"))
    if not found:
        raise ReseedError(f"no site-packages under {venv}")
    return found[0]


def _copy(src: Path, dst: Path) -> None:
    """Replace ``dst`` with a copy of ``src`` (a file or a whole tree)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".reseed-tmp")
    old = dst.with_name(dst.name + ".reseed-old")
    for p in (tmp, old):
        if p.is_dir() and not p.is_symlink():
            shutil.rmtree(p)
        elif p.exists() or p.is_symlink():
            p.unlink()
    r = subprocess.run(["cp", "-a", "--reflink=auto", str(src), str(tmp)], capture_output=True, text=True)
    if r.returncode != 0:
        raise ReseedError(f"copying {src} failed (is the seed being rebuilt?): {r.stderr.strip()[-300:]}")
    if dst.exists() or dst.is_symlink():
        dst.rename(old)
    tmp.rename(dst)
    if old.is_dir() and not old.is_symlink():
        shutil.rmtree(old)
    elif old.exists() or old.is_symlink():
        old.unlink()


def _artifacts(seed_src: Path, wt: Path) -> list[str]:
    """The seed's gitignored files outside build/ (whole dirs where a dir is
    all ignored) that aren't tracked in the job."""
    out = _git(seed_src, "ls-files", "-z", "--others", "--ignored", "--exclude-standard", "--directory").stdout
    entries = [e for e in out.split("\0") if e and not e.startswith((".", "build/"))
               and "__pycache__" not in e and not e.rstrip("/").endswith(".egg-info")]
    if not entries:
        return []
    tracked = set(_git(wt, "ls-files", "-z", "--", *entries, check=False).stdout.split("\0"))
    return [e for e in entries if e.rstrip("/") not in tracked and not any(t.startswith(e) for t in tracked if t)]


def sync_triton(seed_venv: Path, job_venv: Path, log) -> None:
    """Give the job the seed's triton if it has a different one. The seed
    reinstalls triton for its commit every night; jobs never build it."""
    seed_sp, job_sp = _site_packages(seed_venv), _site_packages(job_venv)
    seed_d, job_d = sorted(seed_sp.glob("triton-*.dist-info")), sorted(job_sp.glob("triton-*.dist-info"))
    if not seed_d or [d.name for d in seed_d] == [d.name for d in job_d]:
        return
    for d in job_d:
        shutil.rmtree(d)
    _copy(seed_sp / "triton", job_sp / "triton")
    _copy(seed_d[0], job_sp / seed_d[0].name)
    # Its console scripts (proton, ...): RECORD lists them under ../../../bin/.
    for line in (seed_d[0] / "RECORD").read_text().splitlines():
        rel = line.split(",")[0]
        src = seed_venv / "bin" / rel.rsplit("/", 1)[-1]
        if rel.startswith("../../../bin/") and src.is_file():
            dst = job_venv / "bin" / src.name
            dst.write_text(src.read_text().replace(str(seed_venv.resolve()), str(job_venv.resolve()))
                           .replace(str(seed_venv), str(job_venv.resolve())))
            dst.chmod(0o755)
    log(f"copied the seed's triton ({seed_d[0].name[:-len('.dist-info')]}; the job had "
        f"{', '.join(d.name[:-len('.dist-info')] for d in job_d) or 'none'})")


def _rewrite(path: Path, pairs: list[tuple[str, str]]) -> None:
    text = path.read_text()
    for old, new in pairs:
        text = text.replace(old, new)
    path.write_text(text)


def reseed(seed: Path, seed_venv: Path, wt: Path, job_venv: Path, force: bool = False,
           log=lambda m: print(m, flush=True)) -> str:
    """Copy the seed's compiled torch into the job. Returns the commit."""
    seed_sha = built_commit(seed)
    if not seed_sha:
        raise ReseedError(f"the seed ({seed}) has no build right now (being rebuilt?); try again later")
    head = _git(seed, "rev-parse", "HEAD", check=False).stdout.strip()
    if head != seed_sha:
        raise ReseedError(f"the seed's checkout ({head[:12]}) doesn't match its build ({seed_sha[:12]}); "
                          "it is probably being rebuilt, try again later")
    if _git(wt, "merge-base", "--is-ancestor", seed_sha, "HEAD", check=False).returncode != 0:
        raise ReseedError(f"HEAD doesn't contain the seed's commit {seed_sha[:12]}: move your branch onto it "
                          "first (it is fetch_origin's `base`), or the seed's build won't match your source")
    if compiled := changed_since(wt, seed_sha):
        log(f"note: since the seed's build, {len(compiled)} file(s) that feed the compiled build changed (in your "
            f"commits, or upstream if your base is newer than the seed), so after this, rebuild for them: {', '.join(compiled[:10])}{' ...' if len(compiled) > 10 else ''}")
    seed_sp, job_sp = _site_packages(seed_venv), _site_packages(job_venv)
    if not (seed_sp / FINDER[0]).exists():
        raise ReseedError(f"the seed venv has no {FINDER[0]} (not a scikit-build-core editable install)")
    if built_commit(wt) == seed_sha and not force:
        log(f"the job already has the seed's build ({seed_sha[:12]}); checking it imports")
    else:
        log(f"copying the seed's compiled torch ({seed_sha[:12]}) into {wt} ...")
        _copy(seed / "build" / "install", wt / "build" / "install")
        arts = _artifacts(seed, wt)
        for e in arts:
            _copy(seed / e.rstrip("/"), wt / e.rstrip("/"))
        for d in job_sp.glob("torch-*.dist-info"):
            shutil.rmtree(d)
        for f in FINDER:
            (job_sp / f).unlink(missing_ok=True)
        for p in [*(seed_sp / f for f in FINDER), *seed_sp.glob("torch-*.dist-info")]:
            _copy(p, job_sp / p.name)
        pairs = [(str(seed.resolve()), str(wt.resolve())), (str(seed), str(wt.resolve())),
                 (str(seed_venv.resolve()), str(job_venv.resolve())), (str(seed_venv), str(job_venv.resolve()))]
        for f in [*(job_sp / f for f in FINDER), *job_sp.glob("torch-*.dist-info/*")]:
            if f.is_file() and f.name in (*FINDER, "direct_url.json", "RECORD"):
                _rewrite(f, pairs)
        for pyc in job_sp.glob("__pycache__/_editable_skbc_torch*.pyc"):
            pyc.unlink()
        log(f"copied build/install, {len(arts)} generated path(s), and the editable finder")
    sync_triton(seed_venv, job_venv, log)
    r = subprocess.run([str(job_venv / "bin" / "python"), "-c", "import torch; print(torch.version.git_version)"],
                       capture_output=True, text=True, cwd="/tmp", timeout=600, stdin=subprocess.DEVNULL)
    got = r.stdout.strip().splitlines()[-1:] or [""]
    if r.returncode != 0 or got[0] != seed_sha:
        raise ReseedError(f"torch doesn't import at {seed_sha[:12]} after the copy (got {got[0][:12] or 'an error'}): "
                          f"{r.stderr.strip()[-500:]}")
    if built_commit(seed) != seed_sha:
        raise ReseedError("the seed changed during the copy (a rebuild started); run reseed again later")
    log(f"done: the job's torch is the seed's build of {seed_sha[:12]}; Python-only changes need no rebuild")
    return seed_sha


def auto(seed: Path, seed_venv: Path, wt: Path, job_venv: Path, log=lambda m: print(m, flush=True)) -> int:
    """rebuild.sh's check: 0 if the job's torch matches its source (after
    copying a newer seed build if that is all it takes), 3 if it needs a build."""
    seed_sha, job_sha = built_commit(seed), built_commit(wt)
    # The seed first: copying its build is cheap and also brings generated
    # files (stubs) up to date.
    for sha in dict.fromkeys(s for s in (seed_sha, job_sha) if s):
        if _git(wt, "merge-base", "--is-ancestor", sha, "HEAD", check=False).returncode != 0:
            continue
        if changed_since(wt, sha):
            continue
        if sha != job_sha:
            log(f"reseed: your source is the seed's build commit {sha[:12]} plus changes that don't need "
                "compiling; copying the seed's build instead of rebuilding")
        if sha == seed_sha:
            reseed(seed, seed_venv, wt, job_venv, log=log)  # also brings the seed's triton
        return 0
    return 3


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="reseed.py", description=__doc__.split("\n\n")[0])
    ap.add_argument("worktree", type=Path)
    ap.add_argument("--force", action="store_true", help="copy even if the job already has the seed's build")
    ap.add_argument("--auto", action="store_true", help="rebuild.sh mode (see the docstring)")
    ap.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parent.parent)
    a = ap.parse_args(argv)
    wt, ws = a.worktree, a.workspace
    try:
        if a.auto:
            return auto(ws / "pytorch", ws / ".venv", wt, wt.parent / ".venv")
        reseed(ws / "pytorch", ws / ".venv", wt, wt.parent / ".venv", a.force)
    except (ReseedError, subprocess.SubprocessError, OSError) as e:
        print(f"reseed: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
