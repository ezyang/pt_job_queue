from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from ptq.application import review_service
from ptq.application.review_service import (
    ReviewTarget,
    plan_reviews,
    recommendation,
    review_prompt,
)
from ptq.application.stack_service import StackCommit, StackStatus
from ptq.domain.models import JobRecord, SubmissionMode
from ptq.infrastructure.job_repository import JobRepository


def test_prompt_is_the_bots_command_scoped_to_the_target():
    target = ReviewTarget("abc x", "abc~1", "abc", "Described by HEAD.")
    prompt = review_prompt(target)
    assert prompt.startswith("/pr-review branch\n")
    assert "`git diff abc~1 HEAD`" in prompt
    assert "Described by HEAD." in prompt


def test_stack_job_reviews_each_commit(tmp_path: Path):
    repo = JobRepository(tmp_path / "jobs.json")
    repo.save(JobRecord(job_id="j", workspace="/ws", submission_mode=SubmissionMode.GHSTACK))
    status = StackStatus(
        branch="b", base="main", remote="origin", base_ref="origin/main", dirty=False,
        has_merges=False,
        commits=(StackCommit("a" * 40, "first"), StackCommit("b" * 40, "second")),
    )
    with patch.object(review_service, "inspect_stack", return_value=status):
        targets = plan_reviews(repo, "j")
    assert [(t.base, t.head) for t in targets] == [("a" * 40 + "~1", "a" * 40),
                                                   ("b" * 40 + "~1", "b" * 40)]
    assert all("commit message" in t.description for t in targets)


def test_recommendation_parses_the_skills_verdict():
    assert recommendation("### Recommendation\n**Request Changes**\n\nfix it") == "Request Changes"
    assert recommendation("### Recommendation\n**Approve**\n") == "Approve"
    assert recommendation("no verdict") == "unknown"
