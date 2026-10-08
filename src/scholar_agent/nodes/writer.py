"""Writer: turns the researchers' notes into a cited literature review."""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from scholar_agent import prompts
from scholar_agent.state import ResearchState


def build_writer_input(state: ResearchState) -> str:
    notes = "\n\n".join(
        f"### Sub-question: {f['sub_question']}\n{f['summary']}" for f in state["findings"]
    )
    paper_list = "\n".join(
        f"- [{p['id']}] {p['title']} ({p['published'][:4]})" for p in state["papers"].values()
    )
    text = (
        f"Research question: {state['question']}\n\n"
        f"## Research notes\n\n{notes}\n\n"
        f"## Papers you may cite\n\n{paper_list}"
    )
    if state.get("issues"):
        issues = "\n".join(f"- {i}" for i in state["issues"])
        text += "\n\n" + prompts.WRITER_REVISION.format(issues=issues, draft=state["draft"])
    return text


def make_writer(llm: BaseChatModel):
    def writer(state: ResearchState) -> dict:
        response = llm.invoke(
            [SystemMessage(prompts.WRITER), HumanMessage(build_writer_input(state))]
        )
        return {"draft": response.text, "drafts_written": state.get("drafts_written", 0) + 1}

    return writer
