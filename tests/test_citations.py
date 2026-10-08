from scholar_agent.citations import extract_citations, render_references, unknown_citations

PAPERS = {
    "2401.00001": {
        "id": "2401.00001",
        "title": "Paper One",
        "authors": "A. Author",
        "published": "2024-01-02",
        "url": "http://arxiv.org/abs/2401.00001v1",
        "abstract": "...",
    }
}


def test_extract_handles_versions_groups_and_old_ids():
    text = "A [2401.00001v2]. B [2312.12345, hep-th/9901001]. Again [2401.00001]."
    assert extract_citations(text) == ["2401.00001", "2312.12345", "hep-th/9901001"]


def test_extract_ignores_non_citation_brackets():
    assert extract_citations("See [1] and [this link](http://x) and [Smith et al.]") == []


def test_unknown_citations():
    assert unknown_citations("[2401.00001] [9999.99999]", PAPERS) == ["9999.99999"]


def test_render_references_only_lists_known_cited_papers():
    refs = render_references("Claim [2401.00001]. Fake [9999.99999].", PAPERS)
    assert "Paper One" in refs and "(2024)" in refs
    assert "9999.99999" not in refs


def test_render_references_empty_when_nothing_cited():
    assert render_references("No citations here.", PAPERS) == ""
