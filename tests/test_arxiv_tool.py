from datetime import datetime

import arxiv
import pytest

from scholar_agent.tools.arxiv_search import format_paper, search_papers, to_paper


def make_result(n_authors: int = 2) -> arxiv.Result:
    return arxiv.Result(
        entry_id="http://arxiv.org/abs/2401.01234v1",
        published=datetime(2024, 1, 3),
        title="Reducing Hallucination\n  in RAG",
        authors=[arxiv.Result.Author(f"Author {i}") for i in range(n_authors)],
        summary="We study\n  hallucination   in RAG.",
    )


class FakeClient:
    def __init__(self, results):
        self.results_list = results
        self.last_search = None

    def results(self, search):
        self.last_search = search
        return iter(self.results_list)


def test_to_paper_strips_version_and_whitespace():
    p = to_paper(make_result())
    assert p["id"] == "2401.01234"
    assert p["title"] == "Reducing Hallucination in RAG"
    assert p["abstract"] == "We study hallucination in RAG."
    assert p["published"] == "2024-01-03"


def test_to_paper_truncates_long_author_lists():
    assert to_paper(make_result(n_authors=6))["authors"].endswith("et al.")


def test_format_paper_shows_id_first():
    assert format_paper(to_paper(make_result())).startswith("[2401.01234] Reducing")


def test_search_papers_returns_text_and_structured_papers():
    client = FakeClient([make_result()])
    text, papers = search_papers("rag hallucination", max_results=3, client=client)
    assert client.last_search.query == "rag hallucination"
    assert client.last_search.max_results == 3
    assert "Reducing Hallucination" in text
    assert [p["id"] for p in papers] == ["2401.01234"]


def test_search_papers_handles_no_results():
    text, papers = search_papers("zzz", client=FakeClient([]))
    assert "No arXiv papers found" in text and papers == []


@pytest.mark.live
def test_live_arxiv_search():
    text, papers = search_papers("attention is all you need", max_results=1)
    assert "arxiv.org/abs/" in text and len(papers) == 1
