"""HTTP API for ScholarAgent.

uv run scholar-api        # http://localhost:8000, interactive docs at /docs

POST /runs                    start a run          -> 202 + run status
GET  /runs/{run_id}           progress / plan / report
POST /runs/{run_id}/decision  approve, edit or cancel the plan of a paused run
"""

import secrets
from contextlib import asynccontextmanager
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field, model_validator

from scholar_agent.service import NotAwaitingApprovalError, RunManager, TooManyRunsError

RunState = Literal["running", "awaiting_approval", "done", "cancelled", "failed", "stopped"]


class StartRun(BaseModel):
    question: str = Field(min_length=5, max_length=500)
    auto_approve: bool = False


class Decision(BaseModel):
    action: Literal["approve", "edit", "cancel"]
    sub_questions: list[str] | None = Field(default=None, max_length=6)

    @model_validator(mode="after")
    def edit_needs_questions(self):
        if self.action == "edit" and not any(q.strip() for q in self.sub_questions or []):
            raise ValueError("an edit needs at least one non-empty sub-question")
        return self


def get_manager(request: Request) -> RunManager:
    return request.app.state.manager


Manager = Annotated[RunManager, Depends(get_manager)]
bearer = HTTPBearer(auto_error=False)


def check_token(
    request: Request, creds: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)]
) -> None:
    token = request.app.state.token
    if token and not (creds and secrets.compare_digest(creds.credentials, token)):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing or wrong API token")


class RunStatus(BaseModel):
    run_id: str
    status: RunState
    question: str | None
    sub_questions: list[str]
    events: list[str]
    report: str | None
    cost_usd: float
    cost_by_node: dict[str, float]
    error: str | None


def create_app(manager: RunManager | None = None, api_token: str | None = None) -> FastAPI:
    """Build the app. Tests pass a ready `manager`; otherwise one is built from settings."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if manager is not None:
            app.state.manager, app.state.token = manager, api_token
            yield
            return
        from pathlib import Path

        from langgraph.checkpoint.sqlite import SqliteSaver

        from scholar_agent.config import get_settings
        from scholar_agent.graph import build_graph

        s = get_settings()
        Path(s.checkpoint_db).parent.mkdir(parents=True, exist_ok=True)
        with SqliteSaver.from_conn_string(s.checkpoint_db) as checkpointer:
            graph = build_graph(checkpointer=checkpointer, require_approval=True)
            app.state.manager = RunManager(graph, s.max_cost_usd, s.max_concurrent_runs)
            app.state.token = s.api_token.get_secret_value() if s.api_token else None
            yield

    app = FastAPI(title="ScholarAgent", version="0.1.0", lifespan=lifespan)

    def found(manager: RunManager, run_id: str) -> dict:
        if (run := manager.status(run_id)) is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"no run with id {run_id!r}")
        return run

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/runs", status_code=202, dependencies=[Depends(check_token)])
    def start_run(body: StartRun, manager: Manager) -> RunStatus:
        try:
            run_id = manager.start(body.question, body.auto_approve)
        except TooManyRunsError as e:
            raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, str(e)) from e
        return RunStatus(**found(manager, run_id))

    @app.get("/runs/{run_id}", dependencies=[Depends(check_token)])
    def get_run(run_id: str, manager: Manager) -> RunStatus:
        return RunStatus(**found(manager, run_id))

    @app.post("/runs/{run_id}/decision", status_code=202, dependencies=[Depends(check_token)])
    def decide(run_id: str, body: Decision, manager: Manager) -> RunStatus:
        found(manager, run_id)
        try:
            manager.decide(run_id, body.model_dump())
        except NotAwaitingApprovalError as e:
            raise HTTPException(status.HTTP_409_CONFLICT, "run is not waiting for approval") from e
        except TooManyRunsError as e:
            raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, str(e)) from e
        return RunStatus(**found(manager, run_id))

    return app


def main() -> None:
    import os

    import uvicorn

    # Local only by default; containers set HOST=0.0.0.0, and hosting platforms set PORT.
    host = os.environ.get("HOST", "127.0.0.1")
    uvicorn.run(create_app(), host=host, port=int(os.environ.get("PORT", "8000")))


if __name__ == "__main__":
    main()
