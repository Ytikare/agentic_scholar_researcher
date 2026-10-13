"""Runs research graphs in the background, for the API: start, check, approve, resume.

A run takes minutes, far longer than an HTTP request should wait, so each one executes in a
worker thread while the API answers status questions from the checkpointer.

The graph always pauses for plan review; for auto-approved runs the service answers
"approve" itself. That way one graph serves both modes, and a run can still be resumed
from the CLI (`scholar --resume <id>`), which uses the same checkpoint database.
"""

import threading
from concurrent.futures import Executor, ThreadPoolExecutor
from dataclasses import dataclass, field
from uuid import uuid4

from langgraph.types import Command

from scholar_agent.costs import CostTracker


@dataclass
class RunInfo:
    """What this process knows about a run beyond its checkpoints. Lost on restart."""

    tracker: CostTracker
    question: str | None = None  # known before the first checkpoint is saved
    auto_approve: bool = False
    running: bool = False
    events: list[str] = field(default_factory=list)
    error: str | None = None


class TooManyRunsError(RuntimeError):
    pass


class NotAwaitingApprovalError(RuntimeError):
    pass


def plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def describe(node: str, update: dict | None) -> str | None:
    """A one-line progress message for a finished node."""
    if node == "planner":
        return f"Planner proposed {plural(len(update['sub_questions']), 'sub-question')}"
    if node == "review_plan":
        return (
            f"Plan approved: {plural(len(update['sub_questions']), 'sub-question')}"
            if update
            else None
        )
    if node == "researcher":
        finding = update["findings"][0]
        return f"Researcher done ({len(update['papers'])} papers): {finding['sub_question']}"
    if node == "writer":
        return f"Writer finished draft {update['drafts_written']}"
    if node == "fact_checker":
        n = len(update["issues"])
        return "Fact-checker approved the draft" if n == 0 else f"Fact-checker found {n} issue(s)"
    if node == "finalize":
        return "Report ready"
    return None


class RunManager:
    def __init__(
        self,
        graph,
        budget_usd: float | None = None,
        max_active: int = 2,
        executor: Executor | None = None,
    ):
        self.graph = graph  # must be built with require_approval=True and a checkpointer
        self.budget_usd = budget_usd
        self.max_active = max_active
        self.executor = executor or ThreadPoolExecutor(max_workers=max_active)
        self.runs: dict[str, RunInfo] = {}
        self._lock = threading.Lock()

    def start(self, question: str, auto_approve: bool = False) -> str:
        run_id = uuid4().hex[:8]
        info = RunInfo(CostTracker(self.budget_usd), question, auto_approve)
        self._launch(run_id, info, {"question": question})
        return run_id

    def decide(self, run_id: str, decision: dict) -> None:
        """Answer the plan review of a paused run, then let it continue."""
        if self.status(run_id)["status"] != "awaiting_approval":
            raise NotAwaitingApprovalError(run_id)
        info = self.runs.get(run_id) or RunInfo(CostTracker(self.budget_usd))
        self._launch(run_id, info, Command(resume=decision))

    def status(self, run_id: str) -> dict | None:
        """The run's state, worked out from its checkpoints plus what this process saw."""
        snapshot = self.graph.get_state({"configurable": {"thread_id": run_id}})
        info = self.runs.get(run_id)
        if not snapshot.values and info is None:
            return None
        values = snapshot.values
        if info and info.running:
            status = "running"
        elif info and info.error:
            status = "failed"
        elif snapshot.interrupts:
            status = "awaiting_approval"
        elif snapshot.next:
            status = "stopped"  # was mid-run when the server stopped; resumable from the CLI
        elif values.get("report"):
            status = "done"
        else:
            status = "cancelled"
        cost = info.tracker.summary() if info else {"usd": 0.0, "by_node": {}}
        return {
            "run_id": run_id,
            "status": status,
            "question": values.get("question") or (info.question if info else None),
            "sub_questions": values.get("sub_questions", []),
            "events": list(info.events) if info else [],
            "report": values.get("report"),
            "cost_usd": cost["usd"],
            "cost_by_node": cost["by_node"],
            "error": info.error if info else None,
        }

    def _launch(self, run_id: str, info: RunInfo, graph_input) -> None:
        with self._lock:
            # Re-checked under the lock so two quick decisions can't resume a run twice.
            if run_id in self.runs and self.runs[run_id].running:
                raise NotAwaitingApprovalError(run_id)
            if sum(r.running for r in self.runs.values()) >= self.max_active:
                raise TooManyRunsError(f"{self.max_active} runs are already in progress")
            info.running, info.error = True, None
            self.runs[run_id] = info
        self.executor.submit(self._execute, run_id, info, graph_input)

    def _execute(self, run_id: str, info: RunInfo, graph_input) -> None:
        config = {
            "configurable": {"thread_id": run_id},
            "callbacks": [info.tracker],
            "run_name": "scholar-api",
            "metadata": {"run_id": run_id},
        }
        try:
            while True:
                for update in self.graph.stream(graph_input, config, stream_mode="updates"):
                    for node, data in update.items():
                        if node != "__interrupt__" and (line := describe(node, data)):
                            info.events.append(line)
                paused = self.graph.get_state(config).interrupts
                if not (paused and info.auto_approve):
                    break
                graph_input = Command(resume={"action": "approve"})
        except Exception as e:  # includes BudgetExceededError; finished steps stay saved
            info.error = str(e) or repr(e)
        finally:
            info.running = False
