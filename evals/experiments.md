# Eval experiments

Each experiment changes one thing and is compared with the baseline on the same
8 questions.

## Baseline

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

### Findings

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

### Next experiments

1. Researcher prompt: search for the seminal papers by name before recent work (targets #2).
   Done, see Experiment 1.
2. Semantic Scholar tool with citation-count ranking (targets #2).
3. Measure judge noise by re-running this baseline unchanged, so improvements can be told
   apart from run-to-run variation.

## Experiment 1: search for foundational papers by title first

Run `20261011-153735`. Change: the researcher prompt now asks for 2-4 searches, starting with
exact-title lookups (`ti:"..."`) of the seminal papers for the sub-question (named from the
model's own knowledge, then confirmed on arXiv), followed by 1-2 keyword searches for recent
work. Only papers arXiv actually returns can be cited, so this adds no hallucination risk.

| question | coverage | groundedness | relevance | clarity | pre-2024 papers cited | drafts | sec |
|---|---|---|---|---|---|---|---|
| diffusion-speed | 4 → 4 | 5 → 5 | 5 → 5 | 4 → 4 | 11 → 15 | 2 → 1 | 90 → 101 |
| gnn-oversmoothing | 3 → **4** | 4 → 4 | 4 → 5 | 4 → 4 | 4 → 14 | 1 → 1 | 66 → 111 |
| llm-agents-eval | 3 → **4** | 5 → 4 | 4 → 5 | 5 → 5 | 4 → 7 | 1 → 2 | 79 → 128 |
| long-context | 3 → **4** | 5 → 4 | 5 → 5 | 4 → 4 | 3 → 13 | 1 → 1 | 82 → 128 |
| moe-efficiency | 4 → 4 | 4 → 4 | 5 → 5 | 5 → 4 | 3 → 5 | 3 → 1 | 108 → 117 |
| rag-hallucination | 3 → **4** | 5 → 4 | 4 → 5 | 5 → 4 | 2 → 12 | 1 → 3 | 68 → 132 |
| rlhf-alternatives | 3 → **4** | 5 → 4 | 5 → 5 | 5 → 5 | 2 → 6 | 2 → 1 | 93 → 90 |
| vit-vs-cnn | 3 → **4** | 3 → 4 | 4 → 5 | 4 → 5 | 12 → 19 | 3 → 1 | 120 → 126 |
| **mean** | **3.25 → 4.0** | **4.5 → 4.13** | **4.5 → 5.0** | **4.5 → 4.38** | **41 → 91 (total)** | **1.75 → 1.38** | **88 → 117** |

Unchanged: 0 made-up citations, citation precision 1.0. Unresolved issues 0 → 0.125
(one claim in `rag-hallucination` still flagged after 3 drafts).

Tokens for all 8 runs: Haiku 444k/32k → 473k/49k (input/output), Sonnet 238k/44k →
195k/37k. Researchers search more, but fewer revisions mean less writer and fact-checker
work, so total token use stays about the same.

### Takeaways

- **Coverage improved on 6 of 8 questions and got worse on none.** Foundational papers
  cited more than doubled. The judge's "canonical papers missing" complaints are gone
  (e.g. DPO, InstructGPT and Constitutional AI now appear for `rlhf-alternatives`).
- **Groundedness dipped (4 down, 1 up)**. The judge's reasons are small overreaches,
  like dropped qualifiers or slightly stronger wording than the abstract. They are not
  wrong claims. Denser reviews with more papers give more chances for these slips. With
  one run each this is within plausible judge noise, so it is a thing to watch, not a
  conclusion.
- **Cost: about 30% slower** (88 s → 117 s per question) for roughly the same tokens.
- **Caveat:** one run per condition. Run-to-run noise has not been measured yet, so only
  the consistent per-question direction on coverage is treated as a real effect.

**Decision:** keep the change.

## Judge noise check

`scholar-judge-noise` re-graded the saved reports without changing them: the baseline 3
more times, Experiment 1 2 more times. Together with the original grades that gives 4
grades per baseline report and 3 per Experiment 1 report. Any movement is judge noise.

Benchmark mean of each score, one value per grading pass:

| score | baseline (4 passes) | experiment 1 (3 passes) | change in average | verdict |
|---|---|---|---|---|
| coverage | 3.25, 3.25, 3.125, 3.25 | 4.0, 4.0, 4.0 | **+0.78** | real: 6x larger than the noise range |
| relevance | 4.5, 4.375, 4.5, 4.625 | 5.0, 5.0, 5.0 | **+0.5** | real: ranges don't overlap |
| clarity | 4.5, 4.375, 4.375, 4.125 | 4.375, 4.625, 4.5 | +0.16 | noise: ranges overlap |
| groundedness | 4.5, 4.0, 4.375, 4.125 | 4.125, 4.0, 3.875 | **−0.25** | small, likely real |

Groundedness, per question (average over all passes): 5 questions down, 2 unchanged,
1 up. The clearest case is `rag-hallucination`, scored 5 in all 4 baseline passes and 4 in
all 3 Experiment 1 passes.

### Takeaways

- **The coverage and relevance gains are solid.** They are far larger than anything the
  judge does on its own.
- **The groundedness drop is real but smaller than it first looked:** −0.25, not −0.37.
  The single-pass comparison overstated it, because the baseline's first pass (4.5) was
  its highest of four.
- **A single judge pass can move a benchmark mean by up to 0.5.** One pass is fine to
  spot large effects but not small ones. Averaging 2-3 passes per report is the cheap fix.
- **Next:** the groundedness slip comes from denser reviews stretching abstracts a little
  (dropped qualifiers, stronger wording). A writer or fact-checker tweak aimed at that,
  e.g. telling the fact-checker to flag claims that drop a qualifier, is the obvious next
  experiment.
