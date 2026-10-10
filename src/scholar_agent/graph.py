"""The multi-agent research graph.

START -> planner -> review_plan (human approves / edits) --Send x N--> researcher (parallel)
      -> writer -> fact_checker --issues--> writer   (max N drafts)
                               --ok------> finalize -> END
"""

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from scholar_agent.citations import render_references
from scholar_agent.nodes import (
    make_fact_checker,
    make_planner,
    make_researcher,
    make_review_plan,
    make_writer,
)
from scholar_agent.state import ResearchState


def make_route_after_check(max_drafts: int):
    def route_after_check(state: ResearchState) -> str:
        if not state["issues"] or state["drafts_written"] >= max_drafts:
            return "finalize"
        return "writer"

    return route_after_check


def finalize(state: ResearchState) -> dict:
    """Append a references list built from real arXiv metadata, plus any unresolved issues."""
    parts = [state["draft"].strip()]
    if refs := render_references(state["draft"], state["papers"]):
        parts.append(refs)
    if state["issues"]:
        notes = "\n".join(f"- {i}" for i in state["issues"])
        parts.append(f"## Unresolved fact-check notes\n\n{notes}")
    return {"report": "\n\n".join(parts)}


def build_graph(
    research_llm: BaseChatModel | None = None,
    writer_llm: BaseChatModel | None = None,
    max_drafts: int | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    require_approval: bool = False,
):
    """Build and compile the graph.

    Pass LLMs to inject fakes in tests. `require_approval=True` pauses for a human after
    planning, which needs a `checkpointer` to save the paused state.
    """
    if require_approval and checkpointer is None:
        raise ValueError("require_approval=True needs a checkpointer to save the paused run")
    if research_llm is None or writer_llm is None or max_drafts is None:
        from scholar_agent.config import get_settings
        from scholar_agent.llm import get_llm

        s = get_settings()
        research_llm = research_llm or get_llm(s.llm_model)
        writer_llm = writer_llm or get_llm(s.writer_model or s.llm_model)
        max_drafts = max_drafts or s.max_drafts

    builder = StateGraph(ResearchState)
    builder.add_node("planner", make_planner(research_llm))
    # review_plan routes itself with Command(goto=...); `destinations` is only for drawing.
    builder.add_node(
        "review_plan", make_review_plan(require_approval), destinations=("researcher", END)
    )
    builder.add_node("researcher", make_researcher(research_llm))
    builder.add_node("writer", make_writer(writer_llm))
    builder.add_node("fact_checker", make_fact_checker(writer_llm))
    builder.add_node("finalize", finalize)

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "review_plan")
    # The writer waits until *all* parallel researchers have finished.
    builder.add_edge("researcher", "writer")
    builder.add_edge("writer", "fact_checker")
    builder.add_conditional_edges(
        "fact_checker", make_route_after_check(max_drafts), ["writer", "finalize"]
    )
    builder.add_edge("finalize", END)
    return builder.compile(checkpointer=checkpointer)
