import os

import pytest

from tests.fakes import PAPER_1, PAPER_2

# Tests must not send traces to LangSmith, even if .env turns tracing on. load_dotenv()
# never overrides a variable that is already set, so setting it here wins.
os.environ["LANGSMITH_TRACING"] = "false"
# Tests use fake LLMs, so they never need a real key; this makes them run anywhere (e.g. CI)
# and keeps the real key out of them.
os.environ["ANTHROPIC_API_KEY"] = "test-key-not-used"


@pytest.fixture
def fake_arxiv(monkeypatch):
    """Replace the real arXiv call so graph tests are offline and deterministic."""
    monkeypatch.setattr(
        "scholar_agent.tools.arxiv_search.search_papers",
        lambda query, max_results=5: ("[2401.00001] Paper One ...", [PAPER_1, PAPER_2]),
    )
