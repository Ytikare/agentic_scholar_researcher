"""Researcher: a small ReAct agent that investigates one sub-question.

Several of these run in parallel, one per sub-question (see `fan_out` in graph.py).
"""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.errors import GraphRecursionError

from scholar_agent import prompts
from scholar_agent.nodes.react import build_react_agent
from scholar_agent.state import Paper, ResearcherInput
from scholar_agent.tools import TOOLS

# Each think->search round is 2 graph steps, so this allows ~5 searches.
MAX_STEPS = 12


def make_researcher(llm: BaseChatModel):
    agent = build_react_agent(llm, TOOLS, prompts.RESEARCHER)

    def researcher(state: ResearcherInput) -> dict:
        question = state["sub_question"]
        # Stream so that if the agent hits its step limit, we keep what it found so far.
        msgs = []
        try:
            for snapshot in agent.stream(
                {"messages": [HumanMessage(question)]},
                config={"recursion_limit": MAX_STEPS},
                stream_mode="values",
            ):
                msgs = snapshot["messages"]
        except GraphRecursionError:
            pass

        papers: dict[str, Paper] = {
            p["id"]: p
            for m in msgs
            if isinstance(m, ToolMessage) and m.artifact
            for p in m.artifact
        }
        final = msgs[-1] if msgs else None
        if isinstance(final, AIMessage) and not final.tool_calls:
            summary = final.text
        else:
            summary = "(Researcher hit its step limit before writing notes.)"

        return {
            "findings": [{"sub_question": question, "summary": summary}],
            "papers": papers,
        }

    return researcher
