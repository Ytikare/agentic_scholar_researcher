"""Fact-checker: verifies the draft's citations, first with code, then with an LLM."""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from scholar_agent import prompts
from scholar_agent.citations import extract_citations, unknown_citations
from scholar_agent.state import ResearchState, Review


def make_fact_checker(llm: BaseChatModel):
    reviewer = llm.with_structured_output(Review, method="json_schema")

    def fact_checker(state: ResearchState) -> dict:
        draft, papers = state["draft"], state["papers"]
        issues: list[str] = []

        # 1) Deterministic check: every cited id must be a paper we actually retrieved.
        cited = extract_citations(draft)
        if not cited:
            issues.append("The draft has no citations; every claim must cite an arXiv id.")
        for cid in unknown_citations(draft, papers):
            issues.append(f"[{cid}] is not in the paper list; remove it or use a listed paper.")

        # 2) LLM check: do the cited abstracts actually support the claims?
        sources = "\n\n".join(
            f"[{cid}] {papers[cid]['title']}\nAbstract: {papers[cid]['abstract']}"
            for cid in cited
            if cid in papers
        )
        if sources:
            review: Review = reviewer.invoke(
                [
                    SystemMessage(prompts.FACT_CHECKER),
                    HumanMessage(f"## Review\n\n{draft}\n\n## Cited abstracts\n\n{sources}"),
                ]
            )
            for i in review.issues:
                cite = f" [{i.citation}]" if i.citation else ""
                issues.append(f'"{i.claim}"{cite}: {i.problem}')

        return {"issues": issues}

    return fact_checker
