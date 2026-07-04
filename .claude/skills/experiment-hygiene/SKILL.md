---
name: experiment-hygiene
description: Evaluation-hygiene rules for running benchmarks or reporting numbers from this repo — use whenever running run.py with an LLM agent, adding results, or updating LEARNINGS.md/README tables. Encodes the lessons this project exists to demonstrate.
---

# Experiment hygiene

This repo's headline lesson is that "agent beats BO" claims live or die on
evaluation hygiene. Any run whose numbers might be reported must follow these
rules — they are the distilled findings of LEARNINGS.md.

## Before the run

- **Seeds:** ≥ 8 (10 for anything compared to the paper). Never conclude from
  a single seed; the observed IMP@k spread is ±7–13.
- **Metric:** batch protocols (`--batch-size > 1`) are judged on **IMP@k**
  (per-round proposal quality), not final best-so-far — the final saturates
  and barely separates methods.
- **Fair baseline:** always include `classic_bo` (local-penalization batch) in
  the same run. Comparing an agent against a weak or stale baseline is how
  fake wins happen.
- **Cache on:** never pass `--no-cache` for LLM runs; interrupted runs then
  resume for free and re-analysis costs nothing.

## After the run — before believing it

- **Check the fallback line.** The run ends with either
  `[ok] ... (no fallbacks)` or `[!] ... fell back N/M decisions`. Any fallback
  means those picks were EI/heuristic, **not the model** — numbers from such a
  run are contaminated and must not be reported as agentic results. Rerun the
  missing decisions (the cache keeps the completed ones).
- **Audit the reasoning.** Every agentic run writes
  `results/decisions_<tag>.jsonl`. Run
  `uv run scripts/audit_decisions.py results/decisions_<tag>.jsonl`
  and check that stated strategies match behaviour and that overrules of the
  surrogate do not consistently lose yield.
- **Credibility checks for new headline claims:** a cold-start or
  prior-knowledge claim needs the leakage counter-evidence:
  `scripts/leakage_probe.py` (zero-shot), `--anonymize` (name ablation) and
  `--permute-yields` (memorisation probe) — see the README's
  credibility-checks section for how to read them.

## Reporting

- Report **where** an advantage lives (cold start / mid-trajectory / final),
  not just that it exists, and give the multi-seed spread next to every mean.
- New findings go to `LEARNINGS.md` in the same honest register as the
  existing sections: what we expected, what we saw, what would falsify it.
- README tables only change from clean (zero-fallback), ≥10-seed runs whose
  summary JSON is committed in `results/`.
