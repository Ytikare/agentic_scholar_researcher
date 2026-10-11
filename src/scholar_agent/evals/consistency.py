"""How consistent is the fact-checker? Run it N times on the same draft and compare.

uv run scholar-consistency evals/results/<run>/moe-efficiency.json -n 5

Uses a saved eval record, so the (expensive) research is not repeated.
"""

import argparse
import itertools
import json
import statistics
from pathlib import Path

from rich.console import Console

from scholar_agent.citations import extract_citations

console = Console()


def jaccard(a: set, b: set) -> float:
    """Overlap of two sets: 1.0 = identical, 0.0 = nothing in common."""
    return 1.0 if not a and not b else len(a & b) / len(a | b)


def consistency_stats(runs: list[list[str]]) -> dict:
    """Compare the issue lists from repeated fact-checks of the same draft.

    Issues are worded differently each time, so we compare *which citations* got flagged.
    """
    counts = [len(issues) for issues in runs]
    passed = [n == 0 for n in counts]
    flagged = [set(extract_citations(" ".join(issues))) for issues in runs]
    pairs = list(itertools.combinations(range(len(runs)), 2))
    return {
        "runs": len(runs),
        "issue_counts": counts,
        "mean_issues": round(statistics.mean(counts), 2),
        "stdev_issues": round(statistics.stdev(counts), 2) if len(counts) > 1 else 0.0,
        # Share of run pairs that agree on pass/fail (pass = no issues).
        "verdict_agreement": _mean_or_1([passed[i] == passed[j] for i, j in pairs]),
        # Average overlap of the flagged citations between run pairs.
        "flagged_overlap": _mean_or_1([jaccard(flagged[i], flagged[j]) for i, j in pairs]),
    }


def _mean_or_1(values: list) -> float:
    return round(statistics.mean(values), 3) if values else 1.0


def main() -> None:
    from scholar_agent.config import get_settings
    from scholar_agent.llm import get_llm
    from scholar_agent.nodes import make_fact_checker

    p = argparse.ArgumentParser(prog="scholar-consistency", description=__doc__.split("\n")[0])
    p.add_argument("record", help="a per-question JSON file written by scholar-eval")
    p.add_argument("-n", type=int, default=5, help="how many times to run the fact-checker")
    args = p.parse_args()

    state = json.loads(Path(args.record).read_text(encoding="utf-8"))["state"]
    s = get_settings()
    checker = make_fact_checker(get_llm(s.writer_model or s.llm_model))

    runs = []
    for i in range(1, args.n + 1):
        issues = checker({"draft": state["draft"], "papers": state["papers"]})["issues"]
        console.print(f"[bold]Run {i}:[/] {len(issues)} issue(s)")
        for issue in issues:
            console.print(f"[dim]  - {issue}[/]")
        runs.append(issues)

    console.rule("Consistency")
    console.print_json(data=consistency_stats(runs))


if __name__ == "__main__":
    main()
