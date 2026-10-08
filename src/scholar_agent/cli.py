"""Command-line entry point: `uv run scholar "your research question"`."""

import re
import sys
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown

from scholar_agent.graph import build_graph

console = Console()
REPORTS_DIR = Path("reports")


def show_progress(node: str, update: dict) -> None:
    """Print one line per finished node so you can watch the agents work."""
    if node == "planner":
        console.print("[bold cyan]Planner[/] split the question into:")
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


def save_report(question: str, report: str) -> Path:
    slug = re.sub(r"[^a-z0-9]+", "-", question.lower()).strip("-")[:60]
    path = REPORTS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}-{slug}.md"
    REPORTS_DIR.mkdir(exist_ok=True)
    path.write_text(report, encoding="utf-8")
    return path


def main() -> None:
    question = " ".join(sys.argv[1:]) or console.input("[bold]Research question:[/] ")
    graph = build_graph()
    report = ""

    # stream_mode="updates" yields {node_name: what_it_wrote} after each node runs.
    for update in graph.stream({"question": question}, stream_mode="updates"):
        for node, data in update.items():
            show_progress(node, data)
            if node == "finalize":
                report = data["report"]

    console.rule("Report")
    console.print(Markdown(report))
    console.print(f"\n[dim]Saved to {save_report(question, report)}[/]")


if __name__ == "__main__":
    main()
