"""arXiv search tool the agent can call."""

import threading

import arxiv
from langchain_core.tools import tool

from scholar_agent.citations import strip_version
from scholar_agent.state import Paper

# One shared client: it enforces arXiv's "max 1 request / 3 seconds" rule.
_client = arxiv.Client(page_size=20, delay_seconds=3.0, num_retries=3)
# Researchers run in parallel threads; the lock keeps them from hitting arXiv at once.
_lock = threading.Lock()


def to_paper(result: arxiv.Result) -> Paper:
    authors = ", ".join(a.name for a in result.authors[:4])
    if len(result.authors) > 4:
        authors += " et al."
    return Paper(
        id=strip_version(result.get_short_id()),
        title=" ".join(result.title.split()),
        authors=authors,
        published=f"{result.published:%Y-%m-%d}",
        url=result.entry_id,
        abstract=" ".join(result.summary.split()),
    )


def format_paper(p: Paper) -> str:
    return (
        f"[{p['id']}] {p['title']}\n"
        f"Authors: {p['authors']}\n"
        f"Published: {p['published']}\n"
        f"URL: {p['url']}\n"
        f"Abstract: {p['abstract']}"
    )


def search_papers(
    query: str, max_results: int = 5, client: arxiv.Client | None = None
) -> tuple[str, list[Paper]]:
    """Plain function (no LangChain) so it's easy to unit test.

    Returns (text for the LLM, structured papers for the program).
    """
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    with _lock:
        papers = [to_paper(r) for r in (client or _client).results(search)]
    if not papers:
        return f"No arXiv papers found for query: {query!r}", []
    return "\n\n".join(format_paper(p) for p in papers), papers


# content_and_artifact: the LLM sees only the text; the structured papers ride along
# on ToolMessage.artifact so the program can track exactly which papers were retrieved.
@tool(response_format="content_and_artifact")
def search_arxiv(query: str, max_results: int = 5) -> tuple[str, list[Paper]]:
    """Search arXiv for academic papers relevant to a query.

    Use short keyword queries (e.g. "retrieval augmented generation hallucination"),
    not full sentences. Returns each paper's arXiv id, title, authors, date, URL and
    abstract. Call it several times with different queries to cover a topic.

    Args:
        query: Keywords to search for.
        max_results: Number of papers to return (1-10).
    """
    return search_papers(query, max(1, min(max_results, 10)))
