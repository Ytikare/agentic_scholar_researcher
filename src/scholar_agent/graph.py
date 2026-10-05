"""Phase 1: a single research agent that loops between thinking and searching arXiv.

START -> agent --(wants a tool?)--> tools -> agent -> ... -> END
"""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from scholar_agent.tools import TOOLS

SYSTEM_PROMPT = """\
You are a research assistant that answers questions using academic papers from arXiv.

Search arXiv before answering; run several searches with different keywords if the first
results don't cover the question well. Base every claim on the papers you found and cite
them inline by arXiv id, like [2401.01234]. If the papers don't answer something, say so
rather than filling the gap from memory.

Finish with a short "References" list: arXiv id, title, URL."""


def build_graph(llm: BaseChatModel | None = None):
    """Build and compile the agent graph. Pass `llm` to inject a fake model in tests."""
    if llm is None:
        from scholar_agent.llm import get_llm

        llm = get_llm()
    llm_with_tools = llm.bind_tools(TOOLS)

    def agent(state: MessagesState) -> dict:
        response = llm_with_tools.invoke([SystemMessage(SYSTEM_PROMPT), *state["messages"]])
        return {"messages": [response]}

    builder = StateGraph(MessagesState)
    builder.add_node("agent", agent)
    builder.add_node("tools", ToolNode(TOOLS))
    builder.add_edge(START, "agent")
    # tools_condition routes to "tools" if the last message has tool calls, else END.
    builder.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    builder.add_edge("tools", "agent")
    return builder.compile()
