from datetime import datetime

import arxiv
import pytest

from scholar_agent.tools.arxiv_search import format_paper, search_papers


def make_paper(n_authors: int = 2) -> arxiv.Result:
    return arxiv.Result(
        entry_id="http://arxiv.org/abs/2401.01234v1",
        published=datetime(2024, 1, 3),
        title="Reducing Hallucination in RAG",
        authors=[arxiv.Result.Author(f"Author {i}") for i in range(n_authors)],
        summary="We study\n  hallucination   in RAG.",
    )


class FakeClient:
    def __init__(self, papers):
        self.papers = papers
        self.last_search = None

    def results(self, search):
        self.last_search = search
        return iter(self.papers)


def test_format_paper_includes_id_title_and_clean_abstract():
    text = format_paper(make_paper())
    assert "[2401.01234v1] Reducing Hallucination in RAG" in text
    assert "Published: 2024-01-03" in text
    assert "Abstract: We study hallucination in RAG." in text


def test_format_paper_truncates_long_author_lists():
    assert "et al." in format_paper(make_paper(n_authors=6))


def test_search_papers_passes_query_and_limit():
    client = FakeClient([make_paper()])
    out = search_papers("rag hallucination", max_results=3, client=client)
    assert client.last_search.query == "rag hallucination"
    assert client.last_search.max_results == 3
    assert "Reducing Hallucination" in out


def test_search_papers_handles_no_results():
    assert "No arXiv papers found" in search_papers("zzz", client=FakeClient([]))


@pytest.mark.live
def test_live_arxiv_search():
    assert "arxiv.org/abs/" in search_papers("attention is all you need", max_results=1)
