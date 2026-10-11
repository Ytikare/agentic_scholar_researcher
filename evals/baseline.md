# Eval baseline

Run `20261011-135017`, 2026-10-11. All 8 questions in `questions.json`, no failures.

Setup: planner and researchers on `claude-haiku-5-5`, writer and fact-checker on
`claude-sonnet-5-5`, judge on `claude-opus-5-5`; `max_drafts=3`, 5 arXiv results per search.

| question | cite cov | precision | unknown | drafts | unresolved | rel | cov | grd | clr | sec |
|---|---|---|---|---|---|---|---|---|---|---|
| diffusion-speed | 0.426 | 1 | 0 | 2 | 0 | 5 | 4 | 5 | 4 | 90.5 |
| gnn-oversmoothing | 0.362 | 1 | 0 | 1 | 0 | 4 | 3 | 4 | 4 | 65.6 |
| llm-agents-eval | 0.577 | 1 | 0 | 1 | 0 | 4 | 3 | 5 | 5 | 79 |
| long-context | 0.511 | 1 | 0 | 1 | 0 | 5 | 3 | 5 | 4 | 81.6 |
| moe-efficiency | 0.46 | 1 | 0 | 3 | 0 | 5 | 4 | 4 | 5 | 107.6 |
| rag-hallucination | 0.4 | 1 | 0 | 1 | 0 | 4 | 3 | 5 | 5 | 68.1 |
| rlhf-alternatives | 0.411 | 1 | 0 | 2 | 0 | 5 | 3 | 5 | 5 | 92.7 |
| vit-vs-cnn | 0.321 | 1 | 0 | 3 | 0 | 4 | 3 | 3 | 4 | 119.8 |
| **mean** | **0.433** | **1** | **0** | **1.75** | **0** | **4.5** | **3.25** | **4.5** | **4.5** | **88.1** |

Tokens for all 8 runs (agent only, judge excluded):
- claude-haiku-5-5: 444,003 input / 32,079 output
- claude-sonnet-5-5: 238,021 input / 43,648 output

Columns: `cite cov` = share of sentences with a citation; `precision` / `unknown` = cited ids
that were really retrieved / made up; `drafts` = drafts written (fact-checker revisions + 1);
`rel` `cov` `grd` `clr` = judge scores (1–5) for relevance, coverage, groundedness, clarity.

## Findings

**1. Citations are trustworthy.** Zero made-up citations and zero unresolved fact-check
issues across all runs. The fact-checker forced a revision in 4 of 8 runs and every flagged
problem was fixed.

**2. Coverage (3.25) is the weakest score, and the cause is retrieval, not writing.** In
almost every run the judge names well-known foundational papers that are missing:
Switch Transformer / GShard (MoE), AgentBench / GAIA (agents), DPO / Constitutional AI
(RLHF alternatives). arXiv search ranks by keyword match, so it surfaces recent niche
papers instead of the classics, and the researchers can only write about what they find.

**3. The fact-checker has a blind spot for selective citation.** `vit-vs-cnn` passed the
fact-checker after 3 drafts but the judge gave groundedness 3: the review cites a paper for
one point while leaving out its headline finding, which contradicts the review's argument.
The fact-checker only checks that each claim is supported; by design it ignores omissions.

## Next experiments

1. Researcher prompt: search for the seminal papers by name before recent work (targets #2).
2. Semantic Scholar tool with citation-count ranking (targets #2).
3. Measure judge noise by re-running this baseline unchanged, so improvements can be told
   apart from run-to-run variation.
