"""LLM-as-judge: a separate model grades each report against a fixed rubric."""

from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from scholar_agent.citations import extract_citations
from scholar_agent.state import Paper

Score = Literal[1, 2, 3, 4, 5]

JUDGE = """\
You grade literature reviews written by an AI research assistant. You get the research
question, the review, and the abstracts of the papers it cites.

Score each criterion from 1 (very poor) to 5 (excellent):
- relevance: does the review answer the question that was asked?
- coverage: does it cover the main approaches or findings a researcher would expect?
- groundedness: are its claims supported by the cited abstracts? Penalize claims that go
  beyond, or contradict, the abstract they cite.
- clarity: is it well organized and easy to follow?

Write your reasoning first, then the scores. Be strict: 5 means you would not change it."""


class JudgeScore(BaseModel):
    # Reasoning comes first so the model thinks before it commits to numbers.
    reasoning: str = Field(description="Short justification for the scores below")
    relevance: Score
    coverage: Score
    groundedness: Score
    clarity: Score

    def scores(self) -> dict[str, int]:
        return self.model_dump(exclude={"reasoning"})


def make_judge(llm: BaseChatModel):
    grader = llm.with_structured_output(JudgeScore, method="json_schema")

    def judge(
        question: str, draft: str, papers: dict[str, Paper], config: RunnableConfig | None = None
    ) -> JudgeScore:
        sources = "\n\n".join(
            f"[{cid}] {papers[cid]['title']}\nAbstract: {papers[cid]['abstract']}"
            for cid in extract_citations(draft)
            if cid in papers
        )
        return grader.invoke(
            [
                SystemMessage(JUDGE),
                HumanMessage(
                    f"## Question\n\n{question}\n\n## Review\n\n{draft}\n\n"
                    f"## Cited abstracts\n\n{sources or '(none)'}"
                ),
            ],
            config,
        )

    return judge
