"""How noisy is the judge? Re-grade saved reports N times and see how much scores move.

uv run scholar-judge-noise evals/results/<run> [evals/results/<other run> ...] -n 3

Nothing about the reports changes between repeats, so any movement is judge noise. If the
benchmark mean of a score moves by 0.4 on its own, a 0.4 difference between two
experiments proves nothing. Only the judge is re-run here, so this measures judge noise,
not run-to-run variation of the agent itself.
"""

import argparse
import json
import statistics
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rich.console import Console
from rich.table import Table

from scholar_agent.evals.run import SCORES

console = Console()


def noise_stats(scores: dict[str, list[dict[str, int]]]) -> dict:
    """`scores` maps question id -> the judge's scores from each repeat."""
    repeats = min(len(runs) for runs in scores.values())
    stats = {}
    for c in SCORES:
        per_question = [[run[c] for run in runs[:repeats]] for runs in scores.values()]
        # The benchmark mean you would have reported after each single repeat.
        means = [round(statistics.mean(q[i] for q in per_question), 3) for i in range(repeats)]
        stats[c] = {
            # Share of questions that got the exact same score every time.
            "unanimous": round(statistics.mean(len(set(q)) == 1 for q in per_question), 3),
            "avg_question_stdev": round(
                statistics.mean(statistics.pstdev(q) for q in per_question), 3
            ),
            "benchmark_means": means,
            "benchmark_mean_range": round(max(means) - min(means), 3),
        }
    return stats


def load_records(run_dir: Path) -> list[dict]:
    records = [json.loads(f.read_text(encoding="utf-8")) for f in sorted(run_dir.glob("*.json"))]
    return [r for r in records if "state" in r]  # skip summaries, noise files and failed runs


def print_stats(title: str, stats: dict) -> None:
    table = Table(title=title)
    for h in ("score", "unanimous", "avg stdev / question", "benchmark mean per repeat", "range"):
        table.add_column(h, justify="left" if h == "score" else "right")
    for c, s in stats.items():
        table.add_row(
            c,
            f"{s['unanimous']:.0%}",
            f"{s['avg_question_stdev']:g}",
            "  ".join(f"{m:.3g}" for m in s["benchmark_means"]),
            f"{s['benchmark_mean_range']:g}",
        )
    console.print(table)


def main() -> None:
    from scholar_agent.config import get_settings
    from scholar_agent.evals.judge import make_judge
    from scholar_agent.llm import get_llm

    p = argparse.ArgumentParser(prog="scholar-judge-noise", description=__doc__.split("\n")[0])
    p.add_argument("runs", nargs="+", help="result folders written by scholar-eval")
    p.add_argument("-n", type=int, default=3, help="how many times to re-grade each report")
    p.add_argument("--workers", type=int, default=4, help="judge calls to run in parallel")
    args = p.parse_args()

    judge = make_judge(get_llm(get_settings().judge_model))
    for run in args.runs:
        run_dir = Path(run)
        records = load_records(run_dir)
        console.print(f"[bold cyan]{run_dir.name}[/]: re-grading {len(records)} reports x{args.n}")

        def grade(r: dict) -> dict[str, int]:
            s = r["state"]
            return judge(r["question"], s["draft"], s["papers"]).scores()

        jobs = [r for r in records for _ in range(args.n)]
        with ThreadPoolExecutor(args.workers) as pool:
            results = list(pool.map(grade, jobs))

        scores: dict[str, list[dict[str, int]]] = {r["id"]: [] for r in records}
        for r, s in zip(jobs, results, strict=True):
            scores[r["id"]].append(s)
        stats = noise_stats(scores)
        out = {"repeats": args.n, "scores": scores, "stats": stats}
        (run_dir / "judge_noise.json").write_text(json.dumps(out, indent=2), "utf-8")
        print_stats(f"Judge noise: {run_dir.name}", stats)


if __name__ == "__main__":
    main()
