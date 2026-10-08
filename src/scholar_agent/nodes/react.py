"""A reusable ReAct loop (think -> call tool -> think ...), the Phase 1 graph generalized.

START -> agent --(tool calls?)--> tools -> agent -> ... -> END
"""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition


def build_react_agent(llm: BaseChatModel, tools: list[BaseTool], system_prompt: str):
    llm_with_tools = llm.bind_tools(tools)

    def agent(state: MessagesState) -> dict:
        response = llm_with_tools.invoke([SystemMessage(system_prompt), *state["messages"]])
        return {"messages": [response]}

    builder = StateGraph(MessagesState)
    builder.add_node("agent", agent)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    builder.add_edge("tools", "agent")
    return builder.compile()
