"""Command-line entry point.

uv run scholar "your research question"     # new run, asks you to approve the plan
uv run scholar -y "your research question"  # new run, auto-approve the plan
uv run scholar --resume <run-id>            # continue a paused / crashed run
uv run scholar --max-cost 0.5 "..."         # stop if the run would cost more than $0.50
"""

import argparse
import re
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

from scholar_agent.config import get_settings
from scholar_agent.costs import BudgetExceededError, CostTracker
from scholar_agent.graph import build_graph
from scholar_agent.service import describe

console = Console()
REPORTS_DIR = Path("reports")


# Colour of each node's progress line.
STYLES = {"planner": "bold cyan", "review_plan": "bold cyan", "researcher": "green",
          "writer": "magenta", "fact_checker": "yellow", "finalize": "bold green"}  # fmt: skip


def show_progress(node: str, update: dict | None) -> None:
    """Print one line per finished node (shared wording with the API), plus CLI-only details."""
    if node == "review_plan" and not update:
        console.print("Run cancelled.", style="red")
        return
    if line := describe(node, update):
        console.print(line, style=STYLES.get(node), markup=False, highlight=False)
    # LLM-written text is printed with markup off, so brackets in it can't break the output.
    details = {"review_plan": "sub_questions", "fact_checker": "issues"}.get(node)
    for item in update.get(details, []) if details else []:
        console.print(f"  - {item}", style="dim", markup=False, highlight=False)


def ask_plan_decision(payload: dict) -> dict:
    """Show the proposed plan and ask the human what to do with it."""
    console.rule("Review the research plan")
    console.print(f"[bold]Question:[/] {payload['question']}\n")
    for i, q in enumerate(payload["sub_questions"], 1):
        console.print(f"  {i}. {q}")
    while True:
        # "\[" escapes the bracket so rich doesn't read "[a]" as a style tag.
        choice = console.input("\n[bold]\\[a][/]pprove, [bold]\\[e][/]dit, [bold]\\[c][/]ancel? ")
        choice = choice.strip().lower()[:1]
        if choice == "a":
            return {"action": "approve"}
        if choice == "c":
            return {"action": "cancel"}
        if choice == "e":
            return {"action": "edit", "sub_questions": edit_sub_questions(payload["sub_questions"])}


def edit_sub_questions(current: list[str]) -> list[str]:
    console.print("[dim]For each: Enter = keep, '-' = remove, or type a replacement.[/]")
    edited = []
    for i, q in enumerate(current, 1):
        answer = console.input(f"  {i}. {q}\n     > ").strip()
        if answer == "-":
            continue
        edited.append(answer or q)
    console.print("[dim]Add more sub-questions (empty line to finish):[/]")
    while extra := console.input("     + ").strip():
        edited.append(extra)
    return edited


def show_costs(tracker: CostTracker) -> None:
    s = tracker.summary()
    console.print(f"[bold]Cost: ${s['usd']:.3f}[/] over {s['calls']} LLM calls")
    table = Table()
    table.add_column("node")
    table.add_column("USD", justify="right")
    for node, usd in s["by_node"].items():
        table.add_row(node, f"{usd:.4f}")
    console.print(table)
    for model, m in s["by_model"].items():
        console.print(f"[dim]{model}: {m['input']:,} in / {m['output']:,} out = ${m['usd']:.4f}[/]")
    if s["unpriced_models"]:
        console.print(f"[yellow]No price known for {s['unpriced_models']}; not in the total.[/]")


def save_report(question: str, report: str) -> Path:
    slug = re.sub(r"[^a-z0-9]+", "-", question.lower()).strip("-")[:60]
    path = REPORTS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}-{slug}.md"
    REPORTS_DIR.mkdir(exist_ok=True)
    path.write_text(report, encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="scholar", description="Multi-agent arXiv literature review.")
    p.add_argument("question", nargs="*", help="research question")
    p.add_argument("-y", "--yes", action="store_true", help="auto-approve the research plan")
    p.add_argument("--resume", metavar="RUN_ID", help="continue a paused or crashed run")
    p.add_argument("--max-cost", type=float, metavar="USD", help="stop the run above this cost")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    db_path = Path(settings.checkpoint_db)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with SqliteSaver.from_conn_string(str(db_path)) as checkpointer:
        graph = build_graph(checkpointer=checkpointer, require_approval=not args.yes)

        # A thread_id names one run; every step of it is saved under this id.
        run_id = args.resume or uuid4().hex[:8]
        budget = args.max_cost if args.max_cost is not None else settings.max_cost_usd
        tracker = CostTracker(budget)  # counts this session only, not earlier ones of a resume
        config = {
            "configurable": {"thread_id": run_id},
            "callbacks": [tracker],
            # Shown in traces (LangSmith), so a run can be found by its id.
            "run_name": "scholar",
            "metadata": {"run_id": run_id},
        }

        if args.resume:
            snapshot = graph.get_state(config)
            if not snapshot.values:
                console.print(f"[red]No saved run with id {run_id!r}.[/]")
                return
            question = snapshot.values["question"]
            console.print(f"Resuming run [bold]{run_id}[/]: {question}")
            # None = "continue from the last checkpoint" (e.g. after a crash).
            graph_input, pending = None, snapshot.interrupts
        else:
            question = " ".join(args.question) or console.input("[bold]Research question:[/] ")
            console.print(f"Run id: [bold]{run_id}[/]")
            graph_input, pending = {"question": question}, ()

        try:
            while True:
                if pending:
                    # The graph is paused for human review: ask, then resume with the answer.
                    graph_input = Command(resume=ask_plan_decision(pending[0].value))
                # stream_mode="updates" yields {node_name: what_it_wrote} after each node.
                for update in graph.stream(graph_input, config, stream_mode="updates"):
                    for node, data in update.items():
                        if node != "__interrupt__":
                            show_progress(node, data)
                pending = graph.get_state(config).interrupts
                if not pending:
                    break
        except (KeyboardInterrupt, EOFError):
            # Every finished step is already saved, so nothing is lost.
            console.print(f"\n[yellow]Paused.[/] Resume with: uv run scholar --resume {run_id}")
            show_costs(tracker)
            return
        except BudgetExceededError as e:
            console.print(f"\n[red]Stopped: {e}.[/] Finished steps are saved; to continue:")
            console.print(f"  uv run scholar --resume {run_id} --max-cost <higher limit>")
            show_costs(tracker)
            return

        snapshot = graph.get_state(config)

    report = snapshot.values.get("report")
    if report:
        console.rule("Report")
        console.print(Markdown(report))
        console.print(f"\n[dim]Saved to {save_report(question, report)}[/]")
    show_costs(tracker)


if __name__ == "__main__":
    main()
