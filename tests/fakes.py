"""A fake chat model for offline tests.

Instead of a fixed script, it takes functions that decide the reply from the input messages.
That keeps it deterministic even when parallel researchers call it in any order.
"""

from collections.abc import Callable
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda


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
