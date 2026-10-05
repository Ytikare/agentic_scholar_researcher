"""arXiv search tool the agent can call."""

import arxiv
from langchain_core.tools import tool

# One shared client: it enforces arXiv's "max 1 request / 3 seconds" rule.
_client = arxiv.Client(page_size=20, delay_seconds=3.0, num_retries=3)


def format_paper(paper: arxiv.Result) -> str:
    authors = ", ".join(a.name for a in paper.authors[:4])
    if len(paper.authors) > 4:
        authors += " et al."
    abstract = " ".join(paper.summary.split())
    return (
        f"[{paper.get_short_id()}] {paper.title}\n"
        f"Authors: {authors}\n"
        f"Published: {paper.published:%Y-%m-%d}\n"
        f"URL: {paper.entry_id}\n"
        f"Abstract: {abstract}"
    )


def search_papers(query: str, max_results: int = 5, client: arxiv.Client | None = None) -> str:
    """Plain function (no LangChain) so it's easy to unit test."""
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    papers = list((client or _client).results(search))
    if not papers:
        return f"No arXiv papers found for query: {query!r}"
    return "\n\n".join(format_paper(p) for p in papers)


@tool
def search_arxiv(query: str, max_results: int = 5) -> str:
    """Search arXiv for academic papers relevant to a query.

    Use short keyword queries (e.g. "retrieval augmented generation hallucination"),
    not full sentences. Returns each paper's arXiv id, title, authors, date, URL and
    abstract. Call it several times with different queries to cover a topic.

    Args:
        query: Keywords to search for.
        max_results: Number of papers to return (1-10).
    """
    return search_papers(query, max(1, min(max_results, 10)))
