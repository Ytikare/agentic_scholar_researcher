"""Phase 3: pausing for human plan review, and resuming runs from saved checkpoints."""

from collections import Counter

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from scholar_agent.graph import build_graph
from scholar_agent.state import Review
from tests.fakes import research_llm, writer_llm

pytestmark = pytest.mark.usefixtures("fake_arxiv")

DRAFT = "X works well [2401.00001]."
CONFIG = {"configurable": {"thread_id": "test-run"}}


def make_graph(calls: Counter | None = None, fail_once_on: str | None = None, approval=True):
    return build_graph(
        research_llm(calls, fail_once_on),
        writer_llm(DRAFT, [Review(issues=[])]),
        max_drafts=3,
        checkpointer=InMemorySaver(),
        require_approval=approval,
    )


def test_graph_pauses_for_plan_review_before_any_research():
    calls = Counter()
    graph = make_graph(calls)
    graph.invoke({"question": "Does X work?"}, CONFIG)

    snapshot = graph.get_state(CONFIG)
    assert snapshot.interrupts[0].value["sub_questions"] == ["Sub A?", "Sub B?"]
    assert snapshot.next == ("review_plan",)
    assert calls["Sub A?"] == 0  # nothing spent on research yet


def test_approve_resumes_and_finishes():
    graph = make_graph()
    graph.invoke({"question": "Does X work?"}, CONFIG)
    result = graph.invoke(Command(resume={"action": "approve"}), CONFIG)

    assert len(result["findings"]) == 2
    assert "## References" in result["report"]


def test_edited_plan_is_what_gets_researched():
    graph = make_graph()
    graph.invoke({"question": "Does X work?"}, CONFIG)
    edited = ["Only this?", "  ", "And this?"]
    result = graph.invoke(Command(resume={"action": "edit", "sub_questions": edited}), CONFIG)

    assert result["sub_questions"] == ["Only this?", "And this?"]  # blanks dropped
    assert sorted(f["sub_question"] for f in result["findings"]) == ["And this?", "Only this?"]


def test_cancel_stops_without_research():
    calls = Counter()
    graph = make_graph(calls)
    graph.invoke({"question": "Does X work?"}, CONFIG)
    result = graph.invoke(Command(resume={"action": "cancel"}), CONFIG)

    assert "report" not in result and result["findings"] == []
    assert graph.get_state(CONFIG).next == ()  # run is over
    assert calls["Sub A?"] == 0


def test_crashed_run_resumes_without_redoing_finished_work():
    calls = Counter()
    graph = make_graph(calls, fail_once_on="Sub B?", approval=False)

    with pytest.raises(RuntimeError, match="simulated crash"):
        graph.invoke({"question": "Does X work?"}, CONFIG)

    # Resume: passing None means "continue this thread from its last checkpoint".
    result = graph.invoke(None, CONFIG)

    assert "## References" in result["report"]
    assert calls["planner"] == 1  # planning was not repeated
    assert calls["Sub A?"] == 1  # the researcher that succeeded was not re-run
    assert calls["Sub B?"] == 2  # only the crashed researcher ran again
    assert len(result["findings"]) == 2


def test_require_approval_without_checkpointer_is_rejected():
    with pytest.raises(ValueError, match="checkpointer"):
        build_graph(research_llm(), writer_llm(DRAFT, []), max_drafts=1, require_approval=True)
