from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from scholar_agent import graph as graph_module


class FakeToolModel(GenericFakeChatModel):
    """Replays scripted AIMessages; ignores bind_tools so the graph can use it."""

    def bind_tools(self, tools, **kwargs):
        return self


def test_agent_calls_tool_then_answers(monkeypatch):
    # Replace the real arXiv call so the test is offline and deterministic.
    monkeypatch.setattr(
        "scholar_agent.tools.arxiv_search.search_papers",
        lambda query, max_results=5: f"[2401.00001] Fake paper about {query}\nURL: x",
    )
    scripted = iter(
        [
            AIMessage(
                content="",
                tool_calls=[{"name": "search_arxiv", "args": {"query": "rag"}, "id": "call_1"}],
            ),
            AIMessage(content="RAG reduces hallucination [2401.00001]."),
        ]
    )
    graph = graph_module.build_graph(llm=FakeToolModel(messages=scripted))

    result = graph.invoke({"messages": [HumanMessage("What is RAG?")]})
    msgs = result["messages"]

    # human -> ai(tool call) -> tool result -> ai(final answer)
    assert [type(m) for m in msgs] == [HumanMessage, AIMessage, ToolMessage, AIMessage]
    assert "Fake paper about rag" in msgs[2].content
    assert msgs[-1].content == "RAG reduces hallucination [2401.00001]."


def test_agent_answers_directly_without_tools():
    scripted = iter([AIMessage(content="No search needed.")])
    graph = graph_module.build_graph(llm=FakeToolModel(messages=scripted))
    result = graph.invoke({"messages": [HumanMessage("hi")]})
    assert len(result["messages"]) == 2
