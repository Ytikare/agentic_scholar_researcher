"""Human review: pause after planning so a person can approve, edit, or cancel the plan.

`interrupt()` stops the graph and saves its state (needs a checkpointer). The run continues
when someone calls the graph again with `Command(resume=<decision>)`; the decision becomes
the return value of `interrupt()`. On resume this node re-runs from the top, so nothing
with side effects may happen before the `interrupt()` call.

Decision shapes:
    {"action": "approve"}
    {"action": "edit", "sub_questions": ["...", "..."]}
    {"action": "cancel"}
"""

from langgraph.graph import END
from langgraph.types import Command, Send, interrupt

from scholar_agent.state import ResearchState


def fan_out(sub_questions: list[str]) -> list[Send]:
    """One researcher per sub-question. Send() gives each its own input; they run in parallel."""
    return [Send("researcher", {"sub_question": q}) for q in sub_questions]


def make_review_plan(require_approval: bool):
    def review_plan(state: ResearchState) -> Command:
        sub_questions = state["sub_questions"]

        if require_approval:
            decision = interrupt({"question": state["question"], "sub_questions": sub_questions})
            action = decision.get("action")
            if action == "edit":
                sub_questions = [q.strip() for q in decision["sub_questions"] if q.strip()]
            elif action != "approve":
                return Command(goto=END)  # cancelled before spending anything on research

        if not sub_questions:
            return Command(goto=END)
        # Command = "update the state AND choose where to go next" in one return value.
        return Command(update={"sub_questions": sub_questions}, goto=fan_out(sub_questions))

    return review_plan
