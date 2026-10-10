"""Command-line entry point.

uv run scholar "your research question"     # new run, asks you to approve the plan
uv run scholar -y "your research question"  # new run, auto-approve the plan
uv run scholar --resume <run-id>            # continue a paused / crashed run
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

from scholar_agent.config import get_settings
from scholar_agent.graph import build_graph

console = Console()
REPORTS_DIR = Path("reports")


def show_progress(node: str, update: dict | None) -> None:
    """Print one line per finished node so you can watch the agents work."""
    if node == "planner":
        console.print(
            f"[bold cyan]Planner[/] proposed {len(update['sub_questions'])} sub-questions"
        )
    elif node == "review_plan":
        if not update:
            console.print("[red]Run cancelled.[/]")
            return
        console.print("[bold cyan]Plan approved:[/]")
        for q in update["sub_questions"]:
            console.print(f"  • {q}")
    elif node == "researcher":
        f = update["findings"][0]
        console.print(
            f"[green]Researcher[/] done ({len(update['papers'])} papers): {f['sub_question']}"
        )
    elif node == "writer":
        console.print(f"[magenta]Writer[/] finished draft {update['drafts_written']}")
    elif node == "fact_checker":
        n = len(update["issues"])
        console.print(f"[yellow]Fact-checker[/] {'approved' if n == 0 else f'found {n} issue(s)'}")
        for issue in update["issues"]:
            console.print(f"[dim]  - {issue}[/]")


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
    return p.parse_args()


def main() -> None:
    args = parse_args()
    db_path = Path(get_settings().checkpoint_db)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with SqliteSaver.from_conn_string(str(db_path)) as checkpointer:
        graph = build_graph(checkpointer=checkpointer, require_approval=not args.yes)

        # A thread_id names one run; every step of it is saved under this id.
        run_id = args.resume or uuid4().hex[:8]
        config = {"configurable": {"thread_id": run_id}}

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
            return

        snapshot = graph.get_state(config)

    report = snapshot.values.get("report")
    if report:
        console.rule("Report")
        console.print(Markdown(report))
        console.print(f"\n[dim]Saved to {save_report(question, report)}[/]")


if __name__ == "__main__":
    main()
