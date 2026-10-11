"""Metrics computed by code from a finished run. Cheap, exact and repeatable."""

import re

from scholar_agent.citations import extract_citations
from scholar_agent.state import ResearchState

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def sentences(text: str) -> list[str]:
    """Split prose into sentences, skipping headings and blank lines."""
    out = []
    for line in text.splitlines():
        line = line.strip().lstrip("-*0123456789. ").strip()
        if line and not line.startswith("#"):
            out.extend(s for s in _SENTENCE_END.split(line) if len(s.split()) >= 4)
    return out


def report_metrics(state: ResearchState) -> dict:
    draft, papers = state.get("draft", ""), state.get("papers", {})
    sents = sentences(draft)
    cited = extract_citations(draft)
    known = [c for c in cited if c in papers]
    return {
        "words": len(draft.split()),
        # Share of sentences that carry at least one citation.
        "citation_coverage": _ratio(sum(bool(extract_citations(s)) for s in sents), len(sents)),
        "papers_retrieved": len(papers),
        "papers_cited": len(cited),
        # Cited ids that were never retrieved = made-up citations. Should always be 0.
        "unknown_citations": len(cited) - len(known),
        "citation_precision": _ratio(len(known), len(cited)),
        "drafts_written": state.get("drafts_written", 0),
        "unresolved_issues": len(state.get("issues", [])),
    }


def _ratio(a: int, b: int) -> float:
    return round(a / b, 3) if b else 0.0
