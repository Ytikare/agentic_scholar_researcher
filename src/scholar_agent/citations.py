"""Plain-code citation handling: no LLM involved, so it can't hallucinate."""

import re

from scholar_agent.state import Paper

# New-style ids (2401.01234) and old-style ids (hep-th/9901001), optional version suffix.
_ID = re.compile(r"^(\d{4}\.\d{4,5}|[a-z\-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?$")
_BRACKETS = re.compile(r"\[([^\[\]]+)\]")


def strip_version(arxiv_id: str) -> str:
    return re.sub(r"v\d+$", "", arxiv_id)


def extract_citations(text: str) -> list[str]:
    """Return cited arXiv ids in order of first appearance, e.g. from
    "... shown [2401.01234] and [2312.00001v2, 2310.11111]"."""
    seen: dict[str, None] = {}
    for group in _BRACKETS.findall(text):
        for part in re.split(r"[,;]", group):
            m = _ID.match(part.strip())
            if m:
                seen.setdefault(m.group(1), None)
    return list(seen)


def unknown_citations(text: str, papers: dict[str, Paper]) -> list[str]:
    return [c for c in extract_citations(text) if c not in papers]


def render_references(text: str, papers: dict[str, Paper]) -> str:
    lines = []
    for cid in extract_citations(text):
        if p := papers.get(cid):
            lines.append(
                f"- [{cid}] {p['authors']} ({p['published'][:4]}). *{p['title']}*. {p['url']}"
            )
    return "## References\n\n" + "\n".join(lines) if lines else ""
