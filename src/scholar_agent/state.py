"""The data that flows through the graph.

`ResearchState` is the shared "whiteboard" every node reads from and writes to.
Fields wrapped in `Annotated[..., reducer]` are *merged* when several nodes write
to them at once (the parallel researchers); plain fields are simply overwritten.
"""

import operator
from typing import Annotated, TypedDict

from pydantic import BaseModel, Field


class Paper(TypedDict):
    id: str  # arXiv id without version, e.g. "2401.01234"
    title: str
    authors: str
    published: str  # YYYY-MM-DD
    url: str
    abstract: str


class Finding(TypedDict):
    sub_question: str
    summary: str  # researcher's cited notes for this sub-question


def merge_papers(left: dict[str, Paper], right: dict[str, Paper]) -> dict[str, Paper]:
    return {**left, **right}


class ResearchState(TypedDict, total=False):
    question: str
    sub_questions: list[str]
    findings: Annotated[list[Finding], operator.add]
    papers: Annotated[dict[str, Paper], merge_papers]
    draft: str
    drafts_written: int
    issues: list[str]  # problems the fact-checker found in the current draft
    report: str


class ResearcherInput(TypedDict):
    """What the planner sends to each parallel researcher."""

    sub_question: str


# --- Structured outputs (Claude must reply in exactly these shapes) ---


class Plan(BaseModel):
    sub_questions: list[str] = Field(
        description="2-4 focused sub-questions that together answer the research question"
    )


class Issue(BaseModel):
    claim: str = Field(description="The problematic sentence or claim, quoted from the draft")
    citation: str = Field(description="The arXiv id it cites, or empty if none")
    problem: str = Field(description="Why the cited abstract does not support the claim")


class Review(BaseModel):
    issues: list[Issue] = Field(description="Empty if every claim is supported")
