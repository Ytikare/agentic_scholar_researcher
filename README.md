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

## Tests

```bash
uv run pytest             # offline unit tests (fake LLM, fake arXiv)
uv run pytest -m live     # tests that hit real APIs
uv run ruff check .       # lint
```

## Roadmap

- [x] Phase 0: project setup (uv, ruff, pytest)
- [x] Phase 1: single ReAct agent with an arXiv tool
- [x] Phase 2: multi-agent graph (planner → parallel researchers → writer → fact-checker)
- [ ] Phase 3: human-in-the-loop + checkpointing
- [ ] Phase 4–5: tests + LLM-as-judge evals
- [ ] Phase 6: tracing & cost tracking
- [ ] Phase 7–8: FastAPI + Streamlit, Docker, CI, deploy
