"""Fake LLMs and data for offline tests.

Instead of a fixed script, FakeLLM takes functions that decide the reply from the input
messages. That keeps it deterministic even when parallel researchers call it in any order.
"""

from collections import Counter
from collections.abc import Callable
from typing import Any
from uuid import uuid4

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda

from scholar_agent.state import Plan, Review

PAPER_1 = {
    "id": "2401.00001",
    "title": "Paper One",
    "authors": "A. Author",
    "published": "2024-01-02",
    "url": "http://arxiv.org/abs/2401.00001",
    "abstract": "X works well.",
}
PAPER_2 = {**PAPER_1, "id": "2401.00002", "title": "Paper Two"}


class FakeLLM(BaseChatModel):
    respond: Callable[[list[BaseMessage]], AIMessage] | None = None
    structured: Callable[[type, list[BaseMessage]], Any] | None = None

    @property
    def _llm_type(self) -> str:
        return "fake"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self.respond(messages))])

    def bind_tools(self, tools, **kwargs):
        return self

    def with_structured_output(self, schema, **kwargs):
        return RunnableLambda(lambda messages: self.structured(schema, messages))


def research_llm(calls: Counter | None = None, fail_once_on: str | None = None) -> FakeLLM:
    """Planner returns 2 sub-questions; each researcher searches once, then writes notes.

    `calls` counts planner runs and researcher starts per sub-question.
    `fail_once_on` makes the researcher for that sub-question crash the first time.
    """
    calls = calls if calls is not None else Counter()

    def respond(messages):
        if any(isinstance(m, ToolMessage) for m in messages):
            return AIMessage("X works well [2401.00001].")
        question = next(m.content for m in messages if isinstance(m, HumanMessage))
        calls[question] += 1
        if question == fail_once_on and calls[question] == 1:
            raise RuntimeError("simulated crash (e.g. network error)")
        call = {"name": "search_arxiv", "args": {"query": "x"}, "id": f"call_{uuid4().hex}"}
        return AIMessage("", tool_calls=[call])

    def structured(schema, messages):
        calls["planner"] += 1
        return Plan(sub_questions=["Sub A?", "Sub B?"])

    return FakeLLM(respond=respond, structured=structured)


def writer_llm(draft: str, reviews: list[Review], seen_inputs: list[str] | None = None):
    """Writer always returns `draft`; fact-checker returns `reviews` in order."""
    reviews_iter = iter(reviews)

    def respond(messages):
        if seen_inputs is not None:
            seen_inputs.append(messages[-1].content)
        return AIMessage(draft)

    return FakeLLM(respond=respond, structured=lambda schema, msgs: next(reviews_iter))
