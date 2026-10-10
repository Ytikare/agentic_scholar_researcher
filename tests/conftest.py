import pytest

from tests.fakes import PAPER_1, PAPER_2


@pytest.fixture
def fake_arxiv(monkeypatch):
    """Replace the real arXiv call so graph tests are offline and deterministic."""
    monkeypatch.setattr(
        "scholar_agent.tools.arxiv_search.search_papers",
        lambda query, max_results=5: ("[2401.00001] Paper One ...", [PAPER_1, PAPER_2]),
    )
