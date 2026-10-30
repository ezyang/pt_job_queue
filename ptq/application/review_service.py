"""``ptq review``: run pytorch/pytorch's automated PR review locally, before submitting.

pytorch/pytorch reviews every PR with Claude Code running the repo's
``/pr-review`` skill (``.github/workflows/hardened-pr-review-run.yml``) and
requests changes on what it finds. This runs the same command, model and
effort on a job's local commits, so its findings can be fixed before the PR
goes up: one review per commit for a ghstack job (each is its own PR), one
for the whole change otherwise. Each review runs in a detached worktree at
the reviewed commit and is written to ``<job>/reviews/<sha>.md``.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from ptq.application.stack_service import (
    checked_git,
    ghstack_remote,
    inspect_stack,
    resolve_base_ref,
)
from ptq.domain.models import PtqError, SubmissionMode
from ptq.infrastructure.backends import backend_for_job
from ptq.infrastructure.job_repository import JobRepository
from ptq.takeover import job_dir_path, worktree_path

# Matches REVIEW_MODEL and `--effort` in hardened-pr-review-run.yml.
REVIEW_MODEL = "claude-opus-5-5"
REVIEW_EFFORT = "high"
REVIEW_TIMEOUT_S = 3000
_RECOMMENDATION_RE = re.compile(r"(?im)^\*\*(approve|request changes|needs discussion)\*\*")


@dataclass(frozen=True)
class ReviewTarget:
    """A change to review as one PR: ``base..head``, described by ``description``."""

    label: str
    base: str
    head: str
    description: str


@dataclass(frozen=True)
class ReviewResult:
    target: ReviewTarget
    path: Path
    recommendation: str
    error: str = ""


def review_prompt(target: ReviewTarget) -> str:
    return (
        "/pr-review branch\n\n"
        "Review this checkout as if it were a pull request: wherever the skill runs "
        "`git diff main...HEAD` (or its `--name-only` / `--stat` forms), use "
        f"`git diff {target.base} HEAD` instead. {target.description} "
        "There is no GitHub conversation yet, so skip the maintainer-comments check. "
        "Do not edit, commit, push or post anything. Your final message is the whole "
        "review in the skill's output format, with each finding marked blocking or "
        "non-blocking and anchored to file:line."
    )


def plan_reviews(repo: JobRepository, job_id: str) -> list[ReviewTarget]:
    job = repo.get(job_id)
    if job.submission_mode == SubmissionMode.GHSTACK:
        status = inspect_stack(repo, job_id)
        return [
            ReviewTarget(
                label=f"{c.sha[:10]} {c.subject}",
                base=f"{c.sha}~1",
                head=c.sha,
                description="The pull request's title and description are HEAD's "
                "commit message (`git log -1 --format=%B`).",
            )
            for c in status.commits
        ]
    backend = backend_for_job(job)
    worktree = worktree_path(backend.workspace, job.job_id, job.repo)
    base_ref = resolve_base_ref(
        backend, worktree, job.stack_base, ghstack_remote(backend, worktree)
    )
    base = checked_git(backend, worktree, f"merge-base {shlex.quote(base_ref)} HEAD").strip()
    head = checked_git(backend, worktree, "rev-parse HEAD").strip()
    if base == head:
        return []
    job_dir = job_dir_path(backend.workspace, job.job_id)
    return [
        ReviewTarget(
            label=f"{head[:10]} (whole change)",
            base=base,
            head=head,
            description=f"The pull request's title is in `{job_dir}/pr_title.txt` and "
            f"its description is built from `{job_dir}/report.md`; read them if they "
            "exist, otherwise use the commit messages.",
        )
    ]


def recommendation(review: str) -> str:
    m = _RECOMMENDATION_RE.search(review)
    return m.group(1).title() if m else "unknown"


def _run_one(worktree: str, out_dir: Path, target: ReviewTarget) -> ReviewResult:
    path = out_dir / f"{target.head[:12]}.md"
    wt = out_dir / f"wt-{target.head[:12]}"
    subprocess.run(["git", "-C", worktree, "worktree", "remove", "--force", str(wt)],
                   capture_output=True)
    try:
        subprocess.run(["git", "-C", worktree, "worktree", "add", "-q", "--detach", str(wt),
                        target.head], check=True, capture_output=True, text=True)
        env = {**os.environ, "META_CLAUDE_CODE_RELEASE": "latest"}
        env.pop("CLAUDE_CODE_VERSION_OVERRIDE", None)
        proc = subprocess.run(
            ["claude", "-p", review_prompt(target), "--model", REVIEW_MODEL,
             "--effort", REVIEW_EFFORT, "--dangerously-skip-permissions"],
            cwd=wt, env=env, capture_output=True, text=True, timeout=REVIEW_TIMEOUT_S,
            stdin=subprocess.DEVNULL,
        )
        path.write_text(proc.stdout)
        if proc.returncode != 0 or not proc.stdout.strip():
            return ReviewResult(target, path, "unknown",
                                (proc.stderr or "no review").strip()[-500:])
        return ReviewResult(target, path, recommendation(proc.stdout))
    except (OSError, subprocess.SubprocessError) as e:
        return ReviewResult(target, path, "unknown", str(e))
    finally:
        subprocess.run(["git", "-C", worktree, "worktree", "remove", "--force", str(wt)],
                       capture_output=True)


def review_job(
    repo: JobRepository,
    job_id: str,
    log: Callable[[str], None] = lambda _: None,
) -> list[ReviewResult]:
    """Review each PR the job would submit, in parallel; the results in stack order."""
    job = repo.get(job_id)
    backend = backend_for_job(job)
    worktree = worktree_path(backend.workspace, job.job_id, job.repo)
    targets = plan_reviews(repo, job_id)
    if not targets:
        raise PtqError("Nothing to review: no commits above the base.")
    out_dir = Path(os.path.expanduser(job_dir_path(backend.workspace, job.job_id))) / "reviews"
    out_dir.mkdir(exist_ok=True)
    for t in targets:
        log(f"Reviewing {t.label}")
    with ThreadPoolExecutor(max_workers=len(targets)) as pool:
        return list(pool.map(lambda t: _run_one(worktree, out_dir, t), targets))
