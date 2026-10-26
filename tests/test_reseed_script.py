import importlib.util
import os
import subprocess
import time
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
_spec = importlib.util.spec_from_file_location("reseed", SCRIPTS / "reseed.py")
reseed = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(reseed)

TAG = "cp312-cp312-linux_aarch64"


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()


def build(src: Path, sha: str, lib: str) -> None:
    """What an editable build leaves behind: build/install plus generated files."""
    t = src / "build" / "install" / TAG / "torch"
    t.mkdir(parents=True, exist_ok=True)
    (t / "version.py").write_text(f"__version__ = '2.16.0a0+git{sha[:7]}'\ngit_version = '{sha}'\n")
    (t / "_C.so").write_text(lib)
    (src / "torch" / "_C").mkdir(parents=True, exist_ok=True)
    (src / "torch" / "_C" / "_VariableFunctions.pyi").write_text(f"# stubs {sha}\n")
    (src / "build" / "CMakeCache.txt").write_text(f"CMAKE_HOME_DIRECTORY={src}\n")


def venv(root: Path, src: Path, sha: str) -> Path:
    v = root / ".venv"
    sp = v / "lib" / "python3.12" / "site-packages"
    (sp / f"torch-2.16.0a0+git{sha[:7]}.dist-info").mkdir(parents=True)
    (sp / f"torch-2.16.0a0+git{sha[:7]}.dist-info" / "direct_url.json").write_text(f'{{"url": "file://{src}"}}')
    (sp / "_editable_skbc_torch.py").write_text(f"install('{src}/build', '{src}/build/install/{TAG}/torch')\n")
    (sp / "_editable_skbc_torch.pth").write_text("import _editable_skbc_torch\n")
    (v / "bin").mkdir()
    # Stands in for `import torch; print(torch.version.git_version)`: report
    # the version.py the finder points at.
    py = v / "bin" / "python"
    py.write_text("#!/bin/sh\nd=$(sed -n \"s/^install('[^']*', '\\([^']*\\)').*/\\1/p\" "
                  f"{sp}/_editable_skbc_torch.py)\nsed -n \"s/^git_version = '\\(.*\\)'/\\1/p\" \"$d/version.py\"\n")
    py.chmod(0o755)
    return v


@pytest.fixture
def world(tmp_path):
    seed = tmp_path / "ws" / "pytorch"
    seed.mkdir(parents=True)
    git(seed, "init", "-q", "-b", "main")
    git(seed, "config", "user.email", "t@t")
    git(seed, "config", "user.name", "t")
    (seed / ".gitignore").write_text("build/\n*.pyi\n")
    (seed / "torch").mkdir()
    (seed / "torch" / "__init__.py").write_text("")
    git(seed, "add", "-A")
    git(seed, "commit", "-qm", "old")
    old = git(seed, "rev-parse", "HEAD")
    job = tmp_path / "job"
    git(tmp_path, "clone", "-q", str(seed), str(job / "pytorch"))
    wt = job / "pytorch"
    git(wt, "config", "user.email", "t@t")
    git(wt, "config", "user.name", "t")
    build(wt, old, "old lib")
    job_venv = venv(job, wt, old)
    # The seed moves on overnight.
    (seed / "torch" / "new.py").write_text("")
    git(seed, "add", "-A")
    git(seed, "commit", "-qm", "new")
    new = git(seed, "rev-parse", "HEAD")
    build(seed, new, "new lib")
    seed_venv = venv(tmp_path / "ws", seed, new)
    return seed, seed_venv, wt, job_venv, old, new


def test_reseed_copies_the_seed_build(world):
    seed, seed_venv, wt, job_venv, _, new = world
    with pytest.raises(reseed.ReseedError, match="move your branch onto"):
        reseed.reseed(seed, seed_venv, wt, job_venv, log=lambda m: None)

    git(wt, "pull", "-q", "--rebase", "origin", "main")
    (wt / "torch" / "mine.py").write_text("x = 1\n")
    git(wt, "add", "-A")
    git(wt, "commit", "-qm", "mine")
    logs = []
    assert reseed.reseed(seed, seed_venv, wt, job_venv, log=logs.append) == new
    inst = wt / "build" / "install" / TAG / "torch"
    assert (inst / "_C.so").read_text() == "new lib"
    assert (wt / "torch" / "_C" / "_VariableFunctions.pyi").read_text() == f"# stubs {new}\n"
    # The job keeps its own CMake tree and its tracked source.
    assert str(wt) in (wt / "build" / "CMakeCache.txt").read_text()
    assert (wt / "torch" / "mine.py").exists()
    sp = job_venv / "lib" / "python3.12" / "site-packages"
    finder = (sp / "_editable_skbc_torch.py").read_text()
    assert str(wt.resolve()) in finder and str(seed) not in finder
    assert [d.name for d in sp.glob("torch-*.dist-info")] == [f"torch-2.16.0a0+git{new[:7]}.dist-info"]
    assert not any("feed the compiled build" in m for m in logs)

    # Again: a no-op. A C++ change of the agent's own is pointed out.
    (wt / "aten").mkdir()
    (wt / "aten" / "x.cpp").write_text("")
    git(wt, "add", "-A")
    git(wt, "commit", "-qm", "c++")
    logs.clear()
    assert reseed.reseed(seed, seed_venv, wt, job_venv, log=logs.append) == new
    assert any("already has the seed's build" in m for m in logs)
    assert any("aten/x.cpp" in m for m in logs)


def test_reseed_refuses_a_seed_mid_rebuild(world):
    seed, seed_venv, wt, job_venv, _, _ = world
    git(wt, "pull", "-q", "--rebase", "origin", "main")
    (seed / "build" / "install" / TAG / "torch" / "version.py").unlink()
    with pytest.raises(reseed.ReseedError, match="no build right now"):
        reseed.reseed(seed, seed_venv, wt, job_venv, log=lambda m: None)


def test_compiled_changes():
    assert reseed.compiled_changes(["torch/a.py", "test/test_x.cpp", "aten/src/ATen/native/native_functions.yaml",
                                    "torch/csrc/x.py", "c10/core/X.h", "CMakeLists.txt", "docs/a.md"]) == [
        "aten/src/ATen/native/native_functions.yaml", "torch/csrc/x.py", "c10/core/X.h", "CMakeLists.txt"]


def test_auto_reseeds_or_asks_for_a_build(world):
    seed, seed_venv, wt, job_venv, _, new = world
    quiet = lambda m: None  # noqa: E731
    # On the old base, with nothing compiled changed: the job's build is current.
    assert reseed.auto(seed, seed_venv, wt, job_venv, log=quiet) == 0
    # Rebased onto the seed: copy its build instead of compiling.
    git(wt, "pull", "-q", "--rebase", "origin", "main")
    (wt / "torch" / "mine.py").write_text("x = 1\n")
    assert reseed.auto(seed, seed_venv, wt, job_venv, log=quiet) == 0
    assert reseed.built_commit(wt) == new
    # An uncommitted C++ change (even untracked) needs a build.
    (wt / "aten").mkdir()
    (wt / "aten" / "x.cpp").write_text("")
    assert reseed.auto(seed, seed_venv, wt, job_venv, log=quiet) == 3


def test_build_slots(tmp_path):
    ws = tmp_path / "ws"
    (ws / "scripts").mkdir(parents=True)
    script = ws / "scripts" / "with-build-slot.sh"
    script.write_bytes((SCRIPTS / "with-build-slot.sh").read_bytes())
    script.chmod(0o755)
    (ws / ".build-slots").mkdir()
    (ws / ".build-slots" / "slot-0").touch()
    (ws / ".build-slots").chmod(0o555)  # sandboxes see it read-only
    try:
        hold = tmp_path / "hold"
        first = subprocess.Popen([str(script), "sh", "-c", f"touch {hold}.1; sleep 3"])
        while not Path(f"{hold}.1").exists():
            time.sleep(0.05)
        status = subprocess.run([str(script), "--status"], capture_output=True, text=True).stdout
        assert status.startswith("slot-0: pid ") and f"touch {hold}.1; sleep 3" in status
        t = time.monotonic()
        second = subprocess.run([str(script), "sh", "-c", "exit 7"], capture_output=True, text=True)
        assert second.returncode == 7  # the command's exit status
        assert subprocess.run([str(script), "sh", "-c", "exit 213"]).returncode == 1  # not "busy"
        assert time.monotonic() - t >= 2  # waited for the only slot
        assert "busy" in second.stderr
        assert first.wait() == 0
        status = subprocess.run([str(script), "--status"], capture_output=True, text=True).stdout
        assert status.strip() == "slot-0: free"
    finally:
        os.chmod(ws / ".build-slots", 0o755)
