# ScholarAgent

A multi-agent LangGraph system that turns a research question into a cited literature review
from arXiv papers, with an automatic fact-checking loop.

## How it works

```
question ─► planner ──Send×N──► researcher ─┐  (parallel ReAct agents searching arXiv)
                                researcher ─┼─► writer ─► fact_checker ──ok──► finalize ─► report
                                researcher ─┘      ▲            │
                                                   └──issues────┘  (max 3 drafts)
```

- **Planner** splits the question into 2–4 sub-questions (structured output).
- **Researchers** run in parallel, each a small tool-using agent that searches arXiv and writes cited notes.
- **Writer** drafts the review, citing papers inline by arXiv id.
- **Fact-checker** verifies citations twice: in code (every cited id must be a retrieved paper,
  so made-up papers can't get through) and with an LLM (does the cited abstract support the claim?).
- **Finalize** builds the reference list from real arXiv metadata, so references can't be made up.

## Quickstart

```bash
uv sync
cp .env.example .env      # then put your ANTHROPIC_API_KEY in .env
uv run scholar "What are current approaches to reducing hallucination in RAG?"
```

The run pauses after planning so you can **approve, edit or cancel** the sub-questions before
any research is done. Every step is saved to a SQLite checkpoint, so a stopped or crashed run
continues where it left off:

```bash
uv run scholar -y "question"          # auto-approve the plan
uv run scholar --resume <run-id>      # continue a paused or crashed run
```

## Tests

```bash
uv run pytest             # offline unit tests (fake LLM, fake arXiv)
uv run pytest -m live     # tests that hit real APIs
uv run ruff check .       # lint
```

## Evals

Unit tests check that the code works; evals check that the **reports are good**.
`evals/questions.json` is a small benchmark of research questions across ML subfields.

```bash
uv run scholar-eval                  # run every question, score each report
uv run scholar-eval --limit 2        # quick smoke run
uv run scholar-consistency evals/results/<run>/<id>.json -n 5
```

Each report is scored two ways:

| Kind | Metric | What it tells you |
|---|---|---|
| Code | `citation_coverage` | share of sentences that cite a paper (strict: one citation per paragraph scores low) |
| Code | `unknown_citations` / `citation_precision` | made-up citations, should always be 0 / 1.0 |
| Code | `drafts_written`, `unresolved_issues` | how much revising the fact-checker forced |
| LLM judge | relevance, coverage, groundedness, clarity (1–5) | quality a researcher would care about; graded by a different, stronger model (`JUDGE_MODEL`) with written reasoning |

Plus latency and tokens per model for every run. `scholar-consistency` re-runs the
fact-checker on one saved draft and reports how often its verdicts agree, because a
checker that flags different things on each run can't be trusted to drive revisions.

Results and experiments so far: [evals/experiments.md](evals/experiments.md).

## Roadmap

- [x] Phase 0: project setup (uv, ruff, pytest)
- [x] Phase 1: single ReAct agent with an arXiv tool
- [x] Phase 2: multi-agent graph (planner → parallel researchers → writer → fact-checker)
- [x] Phase 3: human-in-the-loop + checkpointing
- [x] Phase 4–5: tests + LLM-as-judge evals
- [ ] Phase 6: tracing & cost tracking
- [ ] Phase 7–8: FastAPI + Streamlit, Docker, CI, deploy
