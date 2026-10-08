"""Planner: splits the research question into sub-questions."""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from scholar_agent import prompts
from scholar_agent.state import Plan, ResearchState


def make_planner(llm: BaseChatModel):
    planner_llm = llm.with_structured_output(Plan, method="json_schema")

    def planner(state: ResearchState) -> dict:
        plan: Plan = planner_llm.invoke(
            [SystemMessage(prompts.PLANNER), HumanMessage(state["question"])]
        )
        return {"sub_questions": plan.sub_questions[:4]}

    return planner
