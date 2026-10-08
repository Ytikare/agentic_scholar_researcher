"""End-to-end tests of the multi-agent graph with fake LLMs and a fake arXiv (offline, free)."""

from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from scholar_agent.graph import build_graph
from scholar_agent.state import Issue, Plan, Review
from tests.fakes import FakeLLM

PAPER_1 = {
    "id": "2401.00001",
    "title": "Paper One",
    "authors": "A. Author",
    "published": "2024-01-02",
    "url": "http://arxiv.org/abs/2401.00001",
    "abstract": "X works well.",
}
PAPER_2 = {**PAPER_1, "id": "2401.00002", "title": "Paper Two"}


@pytest.fixture(autouse=True)
def fake_arxiv(monkeypatch):
    monkeypatch.setattr(
        "scholar_agent.tools.arxiv_search.search_papers",
        lambda query, max_results=5: ("[2401.00001] Paper One ...", [PAPER_1, PAPER_2]),
    )


def research_llm() -> FakeLLM:
    """Planner returns 2 sub-questions; each researcher searches once, then writes notes."""

    def respond(messages):
        if any(isinstance(m, ToolMessage) for m in messages):
            return AIMessage("X works well [2401.00001].")
        call = {"name": "search_arxiv", "args": {"query": "x"}, "id": f"call_{uuid4().hex}"}
        return AIMessage("", tool_calls=[call])

    return FakeLLM(
        respond=respond,
        structured=lambda schema, msgs: Plan(sub_questions=["Sub A?", "Sub B?"]),
    )


def writer_llm(draft: str, reviews: list[Review], seen_inputs: list[str] | None = None):
    """Writer always returns `draft`; fact-checker returns `reviews` in order."""
    reviews_iter = iter(reviews)

    def respond(messages):
        if seen_inputs is not None:
            seen_inputs.append(messages[-1].content)
        return AIMessage(draft)

    return FakeLLM(respond=respond, structured=lambda schema, msgs: next(reviews_iter))


def test_happy_path_produces_cited_report():
    graph = build_graph(
        research_llm(), writer_llm("# Review\nX works well [2401.00001].", [Review(issues=[])])
    )
    result = graph.invoke({"question": "Does X work?"})

    assert result["sub_questions"] == ["Sub A?", "Sub B?"]
    assert len(result["findings"]) == 2  # both parallel researchers reported back
    assert set(result["papers"]) == {"2401.00001", "2401.00002"}
    assert result["drafts_written"] == 1
    assert "## References" in result["report"]
    assert "Paper One" in result["report"]
    assert "Paper Two" not in result["report"]  # retrieved but never cited
    assert "Unresolved" not in result["report"]


def test_fact_checker_issues_trigger_a_revision():
    issue = Issue(claim="X works well", citation="2401.00001", problem="overstated")
    seen: list[str] = []
    graph = build_graph(
        research_llm(),
        writer_llm("X works well [2401.00001].", [Review(issues=[issue]), Review(issues=[])], seen),
    )
    result = graph.invoke({"question": "Does X work?"})

    assert result["drafts_written"] == 2
    assert "fact-checker found these problems" in seen[1]
    assert "overstated" in seen[1]
    assert "Unresolved" not in result["report"]


def test_hallucinated_citation_is_caught_and_loop_is_capped():
    draft = "Real [2401.00001]. Made up [9999.99999]."
    graph = build_graph(research_llm(), writer_llm(draft, [Review(issues=[])] * 5), max_drafts=2)
    result = graph.invoke({"question": "Does X work?"})

    assert result["drafts_written"] == 2  # stopped at the cap, not an infinite loop
    references = result["report"].split("## References")[1].split("## Unresolved")[0]
    assert "2401.00001" in references and "9999.99999" not in references
    assert "## Unresolved fact-check notes" in result["report"]
    assert "[9999.99999] is not in the paper list" in result["report"]
