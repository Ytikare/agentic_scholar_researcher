# ScholarAgent

A LangGraph research assistant that searches arXiv and answers research questions with cited papers.
Being built toward a multi-agent literature-review system with evals, observability and deployment.

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
- [ ] Phase 2: multi-agent graph (planner → parallel researchers → writer → fact-checker)
- [ ] Phase 3: human-in-the-loop + checkpointing
- [ ] Phase 4–5: tests + LLM-as-judge evals
- [ ] Phase 6: tracing & cost tracking
- [ ] Phase 7–8: FastAPI + Streamlit, Docker, CI, deploy
