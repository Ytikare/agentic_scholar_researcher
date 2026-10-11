"""Offline tests for the eval tools (metrics, judge, runner, consistency stats)."""

import pytest
from langchain_core.messages import HumanMessage

from scholar_agent.evals.consistency import consistency_stats, jaccard
from scholar_agent.evals.judge import JudgeScore, make_judge
from scholar_agent.evals.judge_noise import noise_stats
from scholar_agent.evals.metrics import report_metrics, sentences
from scholar_agent.evals.run import METRICS, evaluate, markdown_table, summarize
from scholar_agent.graph import build_graph
from scholar_agent.state import Review
from tests.fakes import PAPER_1, PAPER_2, FakeLLM, research_llm, writer_llm

DRAFT = """# Review

## Methods
X works well in most settings [2401.00001]. Some claim Y is better [9999.99999].
This sentence has no citation at all.
- A bullet point that cites a paper [2401.00002].
"""


def test_sentences_skip_headings_and_fragments():
    assert sentences("# Title\n\nShort one. This is a real sentence here.") == [
        "This is a real sentence here."
    ]


def test_report_metrics():
    state = {
        "draft": DRAFT,
        "papers": {p["id"]: p for p in (PAPER_1, PAPER_2)},
        "drafts_written": 2,
        "issues": ["something"],
    }
    m = report_metrics(state)
    assert m["citation_coverage"] == 0.75  # 3 of 4 sentences cite something
    assert m["papers_cited"] == 3
    assert m["unknown_citations"] == 1  # 9999.99999 was never retrieved
    assert m["citation_precision"] == 0.667
    assert m["drafts_written"] == 2 and m["unresolved_issues"] == 1


def test_judge_sees_only_cited_abstracts():
    seen = []
    score = JudgeScore(reasoning="ok", relevance=5, coverage=4, groundedness=3, clarity=5)

    def structured(schema, messages):
        seen.append(next(m.content for m in messages if isinstance(m, HumanMessage)))
        return score

    judge = make_judge(FakeLLM(structured=structured))
    result = judge(
        "Does X work?", "X works [2401.00001].", {"2401.00001": PAPER_1, "2401.00002": PAPER_2}
    )

    assert result.scores() == {"relevance": 5, "coverage": 4, "groundedness": 3, "clarity": 5}
    assert "Paper One" in seen[0] and "Paper Two" not in seen[0]


@pytest.mark.usefixtures("fake_arxiv")
def test_evaluate_runs_graph_and_scores_it():
    graph = build_graph(
        research_llm(), writer_llm("X works well [2401.00001].", [Review(issues=[])])
    )
    score = JudgeScore(reasoning="fine", relevance=4, coverage=3, groundedness=5, clarity=4)

    record = evaluate(graph, lambda q, draft, papers, config=None: score, "Does X work?")

    assert record["metrics"]["citation_precision"] == 1.0
    assert record["judge"]["groundedness"] == 5
    assert "## References" in record["state"]["report"]


def test_summarize_skips_failed_runs():
    ok = {"metrics": {"citation_coverage": 1.0, "citation_precision": 1.0, "unknown_citations": 0,
                      "drafts_written": 1, "unresolved_issues": 0},
          "judge": {"relevance": 4, "coverage": 2, "groundedness": 5, "clarity": 3},
          "seconds": 10.0, "tokens": {"m": {"input": 100, "output": 10}}}  # fmt: skip
    summary = summarize([ok, {**ok, "judge": {**ok["judge"], "coverage": 4}}, {"error": "boom"}])

    assert summary["runs"] == 3 and summary["failed"] == 1
    assert summary["coverage"] == 3
    assert summary["tokens"] == {"m": {"input": 200, "output": 20}}


def test_consistency_stats():
    flag = '"claim" [2401.00001]: overstated'
    stats = consistency_stats([[], [], [flag]])

    assert stats["issue_counts"] == [0, 0, 1]
    assert stats["verdict_agreement"] == 0.333  # only runs 1 and 2 agree
    assert stats["flagged_overlap"] == 0.333
    assert jaccard({"a", "b"}, {"b", "c"}) == pytest.approx(1 / 3)


def test_markdown_table_has_a_row_per_question_and_a_mean():
    record = {"id": "q1", "metrics": dict.fromkeys(METRICS, 1), "seconds": 5.0, "tokens": {}}
    md = markdown_table([record, {"id": "q2", "error": "boom"}], summarize([record]))

    assert md.count("\n| ") == 3  # q1, q2 (error) and mean rows under the header
    assert "error: boom" in md and "| **mean** |" in md


def test_noise_stats():
    def s(cov, grd):
        return {"relevance": 5, "coverage": cov, "groundedness": grd, "clarity": 4}

    stats = noise_stats({"q1": [s(4, 5), s(4, 4)], "q2": [s(3, 5), s(3, 3)]})

    assert stats["coverage"] == {
        "unanimous": 1.0,
        "avg_question_stdev": 0.0,
        "benchmark_means": [3.5, 3.5],
        "benchmark_mean_range": 0.0,
    }
    assert stats["groundedness"]["unanimous"] == 0.0
    assert stats["groundedness"]["benchmark_means"] == [5, 3.5]
    assert stats["groundedness"]["benchmark_mean_range"] == 1.5
