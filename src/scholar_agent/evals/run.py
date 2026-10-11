"""Run the whole agent on a benchmark of questions, then score every report.

uv run scholar-eval                    # all questions, with the LLM judge
uv run scholar-eval --limit 2          # first 2 questions only (cheap smoke run)
uv run scholar-eval --only moe-efficiency --no-judge
"""

import argparse
import json
import statistics
import time
from datetime import datetime
from pathlib import Path

from langchain_core.callbacks import UsageMetadataCallbackHandler
from rich.console import Console
from rich.table import Table

from scholar_agent.evals.metrics import report_metrics

console = Console()
SCORES = ("relevance", "coverage", "groundedness", "clarity")
METRICS = ("citation_coverage", "citation_precision", "unknown_citations", "drafts_written",
           "unresolved_issues")  # fmt: skip


def evaluate(graph, judge, question: str) -> dict:
    """Run one question through the graph and score the result."""
    usage = UsageMetadataCallbackHandler()  # adds up tokens from every LLM call in the run
    start = time.perf_counter()
    state = graph.invoke({"question": question}, {"callbacks": [usage]})
    record = {
        "question": question,
        "seconds": round(time.perf_counter() - start, 1),
        "tokens": {
            model: {"input": u["input_tokens"], "output": u["output_tokens"]}
            for model, u in usage.usage_metadata.items()
        },
        "metrics": report_metrics(state),
    }
    if judge:
        score = judge(question, state["draft"], state["papers"])
        record["judge"] = score.scores()
        record["judge_reasoning"] = score.reasoning
    # Keep the outputs so a run can be inspected, or re-checked later without re-running it.
    record["state"] = {
        k: state.get(k) for k in ("sub_questions", "draft", "papers", "issues", "report")
    }
    return record


def summarize(records: list[dict]) -> dict:
    """Average every metric and judge score over the runs that finished."""
    ok = [r for r in records if "error" not in r]
    summary: dict = {"runs": len(records), "failed": len(records) - len(ok)}
    for key in METRICS:
        summary[key] = _mean([r["metrics"][key] for r in ok])
    for key in SCORES:
        summary[key] = _mean([r["judge"][key] for r in ok if "judge" in r])
    summary["seconds"] = _mean([r["seconds"] for r in ok])
    tokens: dict[str, dict[str, int]] = {}
    for r in ok:
        for model, t in r["tokens"].items():
            total = tokens.setdefault(model, {"input": 0, "output": 0})
            total["input"] += t["input"]
            total["output"] += t["output"]
    summary["tokens"] = tokens
    return summary


def _mean(values: list) -> float | None:
    return round(statistics.mean(values), 3) if values else None


HEADERS = ("question", "cite cov", "precision", "unknown", "drafts", "unresolved",
           "rel", "cov", "grd", "clr", "sec")  # fmt: skip


def table_rows(records: list[dict], summary: dict) -> list[list[str]]:
    """One row per question plus a final `mean` row, shared by the terminal and markdown."""

    def fmt(v):
        return "-" if v is None else f"{v:g}" if isinstance(v, float) else str(v)

    rows = []
    for r in records:
        if "error" in r:
            rows.append([r["id"], f"error: {r['error'][:60]}"] + [""] * (len(HEADERS) - 2))
            continue
        judge = r.get("judge", {})
        rows.append(
            [
                r["id"],
                *(fmt(r["metrics"][k]) for k in METRICS),
                *(fmt(judge.get(k)) for k in SCORES),
                fmt(r["seconds"]),
            ]  # fmt: skip
        )
    rows.append(["mean", *(fmt(summary[k]) for k in (*METRICS, *SCORES, "seconds"))])
    return rows


def token_lines(summary: dict) -> list[str]:
    return [
        f"{model}: {t['input']:,} input / {t['output']:,} output tokens"
        for model, t in summary["tokens"].items()
    ]


def print_table(records: list[dict], summary: dict) -> None:
    table = Table(title="Eval results")
    for i, h in enumerate(HEADERS):
        table.add_column(h, justify="left" if i == 0 else "right")
    *rows, mean = table_rows(records, summary)
    for row in rows:
        table.add_row(*row)
    table.add_section()
    table.add_row(*mean)
    console.print(table)
    for line in token_lines(summary):
        console.print(f"[dim]{line}[/]")


def markdown_table(records: list[dict], summary: dict) -> str:
    rows = table_rows(records, summary)
    rows[-1] = [f"**{v}**" for v in rows[-1]]
    lines = ["| " + " | ".join(HEADERS) + " |", "|" + "---|" * len(HEADERS)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    lines += ["", *(f"- {line}" for line in token_lines(summary))]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="scholar-eval", description="Benchmark ScholarAgent.")
    p.add_argument("--dataset", default="evals/questions.json")
    p.add_argument("--limit", type=int, help="only run the first N questions")
    p.add_argument("--only", nargs="+", metavar="ID", help="only run these question ids")
    p.add_argument("--no-judge", action="store_true", help="skip the LLM judge")
    p.add_argument("--out", default="evals/results")
    return p.parse_args()


def main() -> None:
    from scholar_agent.config import get_settings
    from scholar_agent.evals.judge import make_judge
    from scholar_agent.graph import build_graph
    from scholar_agent.llm import get_llm

    args = parse_args()
    items = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    if args.only:
        items = [i for i in items if i["id"] in args.only]
    items = items[: args.limit]

    graph = build_graph()
    judge = None if args.no_judge else make_judge(get_llm(get_settings().judge_model))
    out_dir = Path(args.out) / f"{datetime.now():%Y%m%d-%H%M%S}"
    out_dir.mkdir(parents=True)

    records = []
    for n, item in enumerate(items, 1):
        console.print(f"[bold cyan]({n}/{len(items)})[/] {item['question']}")
        try:
            record = evaluate(graph, judge, item["question"])
        except Exception as e:  # one failed question shouldn't stop the benchmark
            console.print(f"[red]  failed: {e!r}[/]")
            record = {"question": item["question"], "error": repr(e)}
        record["id"] = item["id"]
        records.append(record)
        (out_dir / f"{item['id']}.json").write_text(json.dumps(record, indent=2), "utf-8")

    summary = summarize(records)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), "utf-8")
    (out_dir / "summary.md").write_text(markdown_table(records, summary), "utf-8")
    print_table(records, summary)
    console.print(f"\n[dim]Saved to {out_dir}[/]")


if __name__ == "__main__":
    main()
