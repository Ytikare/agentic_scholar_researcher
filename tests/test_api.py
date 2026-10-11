"""The HTTP API, end to end with fake LLMs and a fake arXiv (offline, free)."""

from concurrent.futures import Executor, Future

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

from scholar_agent.api import create_app
from scholar_agent.graph import build_graph
from scholar_agent.service import RunManager
from scholar_agent.state import Review
from tests.fakes import research_llm, writer_llm

pytestmark = pytest.mark.usefixtures("fake_arxiv")

QUESTION = {"question": "Does X work well?"}


class InlineExecutor(Executor):
    """Runs submitted work immediately, so tests don't have to wait for threads."""

    def submit(self, fn, *args, **kwargs):
        future = Future()
        future.set_result(fn(*args, **kwargs))
        return future


def client(token=None, max_active=2, fail_once_on=None) -> TestClient:
    graph = build_graph(
        research_llm(fail_once_on=fail_once_on),
        writer_llm("X works well [2401.00001].", [Review(issues=[])] * 3),
        max_drafts=3,
        checkpointer=InMemorySaver(),
        require_approval=True,
    )
    manager = RunManager(graph, max_active=max_active, executor=InlineExecutor())
    return TestClient(create_app(manager, api_token=token))


def test_health():
    with client() as c:
        assert c.get("/health").json() == {"status": "ok", "auth_required": False}


def test_run_pauses_for_approval_then_finishes():
    with client() as c:
        run = c.post("/runs", json=QUESTION).json()
        assert run["status"] == "awaiting_approval"
        assert run["sub_questions"] == ["Sub A?", "Sub B?"]
        assert run["events"] == ["Planner proposed 2 sub-questions"]

        done = c.post(f"/runs/{run['run_id']}/decision", json={"action": "approve"}).json()
        assert done["status"] == "done"
        assert "## References" in done["report"]
        assert "Report ready" in done["events"]
        assert c.get(f"/runs/{run['run_id']}").json()["status"] == "done"


def test_edited_plan_is_researched():
    with client() as c:
        run_id = c.post("/runs", json=QUESTION).json()["run_id"]
        edit = {"action": "edit", "sub_questions": ["Only this?", " "]}
        done = c.post(f"/runs/{run_id}/decision", json=edit).json()

        assert done["sub_questions"] == ["Only this?"]
        assert done["status"] == "done"


def test_cancel():
    with client() as c:
        run_id = c.post("/runs", json=QUESTION).json()["run_id"]
        cancelled = c.post(f"/runs/{run_id}/decision", json={"action": "cancel"}).json()

        assert cancelled["status"] == "cancelled" and cancelled["report"] is None


def test_auto_approve_runs_straight_through():
    with client() as c:
        run = c.post("/runs", json={**QUESTION, "auto_approve": True}).json()

        assert run["status"] == "done"
        assert "Plan approved: 2 sub-questions" in run["events"]


def test_errors():
    with client() as c:
        assert c.get("/runs/nope").status_code == 404
        assert c.post("/runs", json={"question": "Hi"}).status_code == 422  # too short

        run_id = c.post("/runs", json=QUESTION).json()["run_id"]
        empty_edit = {"action": "edit", "sub_questions": ["  "]}
        assert c.post(f"/runs/{run_id}/decision", json=empty_edit).status_code == 422

        c.post(f"/runs/{run_id}/decision", json={"action": "approve"})
        again = c.post(f"/runs/{run_id}/decision", json={"action": "approve"})
        assert again.status_code == 409  # already finished


def test_failed_run_reports_the_error():
    with client(fail_once_on="Sub A?") as c:
        run = c.post("/runs", json={**QUESTION, "auto_approve": True}).json()

        assert run["status"] == "failed"
        assert "simulated crash" in run["error"]


def test_concurrency_limit():
    with client(max_active=0) as c:
        assert c.post("/runs", json=QUESTION).status_code == 429


def test_token_is_required_when_configured():
    with client(token="s3cret") as c:
        assert c.post("/runs", json=QUESTION).status_code == 401
        wrong = {"Authorization": "Bearer nope"}
        assert c.post("/runs", json=QUESTION, headers=wrong).status_code == 401
        right = {"Authorization": "Bearer s3cret"}
        assert c.post("/runs", json=QUESTION, headers=right).status_code == 202
        assert c.get("/health").json()["auth_required"]  # health checks stay open


def test_question_is_known_before_the_first_checkpoint():
    graph = build_graph(research_llm(), writer_llm("x", []), 1, InMemorySaver(), True)

    class NeverRuns(Executor):
        def submit(self, fn, *args, **kwargs):
            return Future()

    with TestClient(create_app(RunManager(graph, executor=NeverRuns()))) as c:
        run = c.post("/runs", json=QUESTION).json()
        assert run["status"] == "running" and run["question"] == QUESTION["question"]
