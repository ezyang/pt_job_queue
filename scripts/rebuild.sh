#!/bin/bash
WORKTREE="${1:?Usage: rebuild.sh <pytorch-worktree-path>}"
JOB_DIR="$(dirname "$WORKTREE")"
JOB_PYTHON="$JOB_DIR/.venv/bin/python"
WORKSPACE="$(dirname "$(dirname "$JOB_DIR")")"
RE_CC_CONFIG="$WORKSPACE/.re-cc-config"
BUILD_ENV="$WORKSPACE/scripts/.build-env"

[ -f "$BUILD_ENV" ] && source "$BUILD_ENV"

# Scope env to the job venv so CMake subprojects (e.g. NNPACK's PeachPy
# codegen) resolve `python` from this venv instead of whatever env
# launched rebuild.sh (uv run, conda, etc.).
unset VIRTUAL_ENV
export PATH="$JOB_DIR/.venv/bin:$PATH"

export CCACHE_NOHASHDIR=true USE_NINJA="${USE_NINJA:-1}" BUILD_TEST="${BUILD_TEST:-0}"

# A rebase onto a newer seed build moves the source past the job's compiled
# torch, and recompiling it is nearly a full build. If the source is a seed
# build plus changes that don't need compiling, copy that build instead (or
# do nothing). PTQ_FORCE_BUILD=1 builds anyway.
if [ "${PTQ_FORCE_BUILD:-0}" != 1 ]; then
    PY="$JOB_PYTHON"
    [ -x "$PY" ] || PY=python3
    "$PY" -I "$WORKSPACE/scripts/reseed.py" --auto "$WORKTREE"
    case $? in
        0) echo "rebuild.sh: torch already matches the source (nothing to compile); PTQ_FORCE_BUILD=1 builds anyway"
           exit 0 ;;
        3) ;;
        *) echo "rebuild.sh: reseed check failed; building" >&2 ;;
    esac
fi

# Fast-path cloned worktrees may still have hardlinked .so files that
# setuptools won't overwrite.  Break them before the editable install.
find "$WORKTREE/torch" -name '*.so' -links +1 \
    -exec cp --remove-destination {} {}.tmp \; \
    -exec mv -f {}.tmp {} \; 2>/dev/null

if [ -f "$RE_CC_CONFIG" ]; then
    MAX_JOBS=$(cat "$RE_CC_CONFIG")
    echo "Using re-cc with MAX_JOBS=$MAX_JOBS"
    cd "$WORKTREE" && MAX_JOBS="$MAX_JOBS" \
        re-cc -- uv pip install --python "$JOB_PYTHON" --no-build-isolation -e . 2>&1
else
    # Builds share the host: cap parallelism (MAX_JOBS above the cap is
    # lowered too), and wait for one of the workspace's build slots.
    CAP=$(( $(nproc) / 8 > 4 ? $(nproc) / 8 : 4 ))
    if [ -z "${MAX_JOBS:-}" ] || [ "$MAX_JOBS" -gt "$CAP" ] 2>/dev/null; then
        export MAX_JOBS="$CAP"
    fi
    echo "Building with MAX_JOBS=$MAX_JOBS"
    cd "$WORKTREE" && \
        "$WORKSPACE/scripts/with-build-slot.sh" nice -n 10 \
        uv pip install --python "$JOB_PYTHON" --no-build-isolation -e . 2>&1
fi
