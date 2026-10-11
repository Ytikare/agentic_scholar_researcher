"""Streamlit front end. It only talks to the API over HTTP, like any other client would.

uv sync --extra ui
uv run scholar-api     # terminal 1: the API on http://localhost:8000
uv run scholar-ui      # terminal 2: this page on http://localhost:8501

SCHOLAR_API_URL and SCHOLAR_API_TOKEN point it at another API (e.g. a deployed one).
"""

import os
import sys
import time

import httpx

API_URL = os.environ.get("SCHOLAR_API_URL", "http://localhost:8000").rstrip("/")
API_TOKEN = os.environ.get("SCHOLAR_API_TOKEN")
POLL_SECONDS = 2
STATUS_LABELS = {
    "running": "🔄 Running",
    "awaiting_approval": "✋ Waiting for your approval",
    "done": "✅ Done",
    "cancelled": "⏹️ Cancelled",
    "failed": "❌ Failed",
    "stopped": "⏸️ Stopped",
}


def api(method: str, path: str, **kwargs) -> dict | None:
    import streamlit as st

    headers = {"Authorization": f"Bearer {API_TOKEN}"} if API_TOKEN else {}
    try:
        r = httpx.request(method, API_URL + path, headers=headers, timeout=30, **kwargs)
    except httpx.HTTPError as e:
        st.error(f"Can't reach the API at {API_URL} ({e}). Is `uv run scholar-api` running?")
        st.stop()
    if r.status_code >= 400:
        detail = r.json().get("detail", r.text) if r.content else r.reason_phrase
        st.error(f"API error {r.status_code}: {detail}")
        return None
    return r.json()


def new_run_form() -> None:
    import streamlit as st

    with st.form("new_run"):
        question = st.text_area(
            "Research question",
            placeholder="What are current approaches to reducing hallucination in RAG?",
            max_chars=500,
        )
        auto = st.checkbox("Skip plan review (approve the research plan automatically)")
        if st.form_submit_button("Start research", type="primary"):
            run = api("POST", "/runs", json={"question": question, "auto_approve": auto})
            if run:
                st.query_params["run"] = run["run_id"]
                st.rerun()


def plan_review(run: dict) -> None:
    import streamlit as st

    st.subheader("Review the research plan")
    st.write("Each line is one sub-question a researcher agent will investigate.")
    edited = st.text_area(
        "Sub-questions (one per line)", "\n".join(run["sub_questions"]), height=160
    )
    questions = [q.strip() for q in edited.splitlines() if q.strip()]
    approve, cancel = st.columns(2)
    if approve.button("Approve and start research", type="primary", use_container_width=True):
        if questions == run["sub_questions"]:
            decision = {"action": "approve"}
        else:
            decision = {"action": "edit", "sub_questions": questions}
        if api("POST", f"/runs/{run['run_id']}/decision", json=decision):
            st.rerun()
    if cancel.button("Cancel run", use_container_width=True):
        if api("POST", f"/runs/{run['run_id']}/decision", json={"action": "cancel"}):
            st.rerun()


def show_run(run: dict) -> None:
    import streamlit as st

    st.markdown(f"**{run['question']}**")
    st.caption(f"Run `{run['run_id']}` · {STATUS_LABELS[run['status']]}")

    if run["events"]:
        with st.expander("Progress", expanded=run["status"] == "running"):
            for line in run["events"]:
                st.write(f"- {line}")

    match run["status"]:
        case "awaiting_approval":
            plan_review(run)
        case "running":
            st.info("The agents are working. This usually takes 1-3 minutes.")
        case "done":
            st.markdown(run["report"])
            st.download_button(
                "Download report (.md)", run["report"], file_name=f"report-{run['run_id']}.md"
            )
        case "failed":
            st.error(run["error"])
            st.write(
                f"Finished steps are saved; resume with `uv run scholar --resume {run['run_id']}`."
            )
        case "stopped":
            st.warning("The server stopped during this run. Finished steps are saved.")
            st.write(f"Resume with `uv run scholar --resume {run['run_id']}`.")
        case "cancelled":
            st.write("This run was cancelled before any research was done.")

    if run["cost_usd"]:
        with st.expander(f"Cost so far: ${run['cost_usd']:.3f}"):
            for node, usd in run["cost_by_node"].items():
                st.write(f"- {node}: ${usd:.4f}")


def page() -> None:
    import streamlit as st

    st.set_page_config(page_title="ScholarAgent", page_icon="📚")
    st.title("📚 ScholarAgent")
    st.caption("Multi-agent literature reviews from arXiv, every claim cited and fact-checked.")

    # The run id lives in the URL, so a refresh or a shared link shows the same run.
    run_id = st.query_params.get("run")
    if not run_id:
        new_run_form()
        return

    if st.button("← New question"):
        del st.query_params["run"]
        st.rerun()
    run = api("GET", f"/runs/{run_id}")
    if run is None:
        return
    show_run(run)

    if run["status"] == "running":
        time.sleep(POLL_SECONDS)  # poll: redraw the page with fresh status
        st.rerun()


def main() -> None:
    """Entry point for `uv run scholar-ui`: start Streamlit on this file."""
    from streamlit.web import cli as stcli

    sys.argv = ["streamlit", "run", __file__, "--server.headless", "true"]
    sys.exit(stcli.main())


if __name__ == "__main__":  # Streamlit runs this file as __main__
    page()
