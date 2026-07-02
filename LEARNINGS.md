# Learnings — Agentic Bayesian Optimisation for reaction/materials discovery

A log of what this small project actually taught us. The headline: getting an
"agent beats classic BO" result to *hold up* is mostly about **honest evaluation** —
the right metric, a fair baseline, and enough seeds — not about the agent.

## Setup in one paragraph

Pool-based Bayesian optimisation: from a finite library of candidates, find the one
with the highest objective in as few (simulated) experiments as possible. Four policies —
**random**, **classic BO** (GP + Expected Improvement), **agentic BO** (an LLM chooses
from a surrogate-informed shortlist), and **agentic BO without the surrogate** (ablation) —
on three datasets of increasing difficulty: a smooth synthetic MOF landscape, the
Buchwald–Hartwig HTE pool (easy: good solutions dense), and Shields et al.'s direct
arylation pool (hard: ~32% of conditions give ~0% yield). Agent backend: Gemini 2.5 Flash
(also Claude), with a deterministic heuristic stand-in for offline runs.

## 1. The metric decides the story

Our first reaction runs showed every method reaching ~85–95% of the optimum — "the agent
doesn't win." That was the *final best-so-far* metric, and it is misleading on a forgiving
pool: with ~33 evaluations from 1728 candidates where ~115 reactions yield ≥70%, even
**random search lands at ~89%**. Best-of-N saturates; it barely separates methods.

The literature ([Reasoning-BO, arXiv:2505.12833](https://arxiv.org/abs/2505.12833)) reports **IMP@k** instead: the *per-round proposal
quality* — the best yield among the batch proposed at round k, **non-cumulative** (so it can
go down). That was the clue: their random baseline is non-monotonic (28.7 → 16.9 → 27.6),
which a best-so-far number can never be. Once we matched the metric, our random reproduced
theirs almost exactly (IMP@1 ≈ 30 vs 29, dip and all) — a validation that setup + metric
were finally aligned.

**Lesson:** pick the metric that discriminates in the regime you care about. For
optimisation under a tight budget, that is early-round behaviour (IMP@k / regret), not the
best point found after the budget saturates.

## 2. Dataset difficulty decides whether the agent helps

The same agent, opposite conclusions on two real datasets (Buchwald numbers from the
per-substrate protocol, `--per-substrate`):

| | Buchwald (easy, saturated) | Arylation (hard, deceptive) |
| --- | --- | --- |
| agentic BO **with** surrogate | **hurts** (86% vs classic 93%) | **helps** (96% vs classic 87%) |
| agentic BO **without** surrogate | ties classic BO, low variance | best cold-start (IMP@1) |

On an easy, dense landscape, greedy EI is already near-optimal, so letting the LLM overrule
it just adds noise — the surrogate + agent *underperforms* plain BO. On a deceptive
landscape (many dead conditions), the surrogate's EI signal and the agent's chemistry prior
**complement** each other and beat plain BO.

**Lesson:** "does an LLM help BO?" has no dataset-independent answer. Benchmark on the
regime that matches your problem; a single easy benchmark will mislead in either direction.

## 3. Reproducing the paper — and the fair-baseline caveat

We aligned the protocol to Reasoning-BO (discrete pool, n_init 3, batch of 3, 30
experiments, 10 seeds, IMP@k metric). Our agentic BO then matched their Reasoning-BO row at
IMP@3 (67.0 vs 66.6) and IMP@5 (71.7 vs 71.2) — a clean reproduction.

But the reproduction only *means* something against a fair baseline. Our first classic BO
used a **greedy top-q EI batch** (pick the 3 highest-EI points), which early on are
near-duplicates in one region → no batch diversity → it gets stuck. Symptom: **random beat
classic BO on the final metric** (89% vs 85%). Scrutinising random exposed this.

Replacing it with a **local-penalization batch** (suppress the acquisition of candidates
near already-chosen points, à la qLogEI) fixed the diversity problem and lifted classic BO's
IMP@5 from 64 → **70** — nearly level with the agent:

| arylation, IMP@k (per-round proposal quality) | IMP@1 | IMP@3 | IMP@5 | final %opt |
| --- | --- | --- | --- | --- |
| random | 30.4 | 25.3 | 48.7 | 89 |
| classic BO — greedy top-q EI (weak) | 48.8 | 61.0 | 64.2 | 85 |
| classic BO — local penalization (fair) | 36.2 | 61.6 | **70.0** | 87 |
| agentic BO — Gemini + surrogate | 41.9 | 67.0 | **71.7** | **96** |
| agentic BO — Gemini, no surrogate | **53.0** | 47.3 | 47.2 | 92 |
| *paper: Vanilla BO* | 43.6 | 45.2 | 55.9 | — |
| *paper: Reasoning-BO* | 60.1 | 66.6 | 71.2 | — |

**Lesson:** an "agent >> baseline" gap is only as trustworthy as the baseline. A weak batch
strategy can manufacture most of the apparent win. (It is worth asking whether the paper's
own Vanilla-BO row, IMP@5 55.9, is a similarly under-tuned batch baseline — a fair BO here
reaches 70.)

## 4. What survives as a real agent advantage

Against the *fair* baseline, the agent's edge is narrower and concentrated:

- **Cold start (IMP@1): real.** The no-surrogate agent scores 53 vs classic BO's 36 — pure
  chemistry knowledge picks good ligand/base/solvent conditions from round one, and this is
  *not* a batch artifact (that policy sees a random shortlist, no EI to exploit).
- **Final / escaping the trap: real.** The agent reaches ~96% while classic BO caps at ~87%
  — on a deceptive landscape EI over-exploits a local region; the agent's diversity of
  reasoning avoids it (visible as the sharp late drop in the regret curve).
- **Mid-trajectory (IMP@5): mostly not.** 71.7 vs a fair 70.0 — the earlier +7.5pp gap was
  largely the weak baseline, not the agent.

**Lesson:** report *where* an advantage lives, not just that it exists. "The agent helps at
cold start and avoids the local-optimum trap, and is otherwise on par with a well-tuned BO"
is more useful — and more credible — than "the agent wins."

## 5. Evaluation hygiene that paid off

- **Multi-seed with visible variance.** The IMP@k gaps sit inside ±7–13 over 10 seeds;
  without the spread we would have over-read single-run noise (as we nearly did with a 3-seed
  "cold-start lead" that turned out to be within the error bars).
- **A persistent decision cache.** Deterministic pipeline + temperature 0 ⇒ prompt→decision
  caching makes interrupted LLM runs resume for free, and re-analysis (new baseline, new plot)
  costs no API calls.
- **Fallback counting.** An early bug had *every* Gemini call 400-ing and silently falling
  back to EI — the "agentic" numbers were the fallback, not the model. Counting fallbacks
  (`[ok] N decisions, 0 fallbacks`) makes a broken agent impossible to mistake for a working
  one.
- **A deterministic heuristic agent.** Lets the whole pipeline (and CI) run offline, and
  gives a knowledge-free floor that isolates how much of the agentic result is *chemistry*
  vs *machinery*.

## Caveats / limitations

- Absolute parity with the paper is not achievable: their search space is under-specified
  (likely continuous concentration/temperature vs our discrete 1728-grid), and our LP batch
  is a stand-in for their qLogEI. Compare *patterns*, not decimals.
- The synthetic MOF objective is physically motivated but not real data.
- IMP@1 for the surrogate-coupled agent (41.9) is below the paper's Reasoning-BO (60.1); its
  first batch spreads wider than theirs. Round-one behaviour is not fully matched.

## What I'd do next

Credibility checks first:

1. **Data-leakage check.** Both datasets are public and plausibly in LLM training data, so
   the cold-start win could be *recall of this dataset* rather than chemistry reasoning.
   Probe it: ask the model zero-shot for the best conditions, and rerun with permuted yields
   to see whether the agent "knows" answers it shouldn't.
2. **Name-ablation prompt.** Strip reagent names from the prompt (anonymous ids or SMILES
   only). If the cold-start advantage disappears, that cleanly proves it comes from named
   chemistry knowledge — the claimed mechanism — and not from something else.
3. **Rationale audit.** Every decision already returns a `strategy` and `rationale` (all
   cached). Check they are connected to reality: do "exploit" rounds yield more than
   "explore" rounds, do picks match their stated reasons, and do the model's chemical claims
   hold against the full ground-truth pool? Good picks with confabulated reasons would point
   back to leakage.

Fairness upgrades to the baseline — the same lesson as §3, on other axes:

4. **BoTorch qLogEI baseline**, to remove any remaining doubt about batch fairness.
5. **Fantasy/penalisation-aware batch for the agent.** Classic BO's batch diversity is
   enforced (local penalization between picks); the agent returns q picks in one shot with
   no such mechanism. Give the agent the same machinery — after each pick, fantasise the
   GP update (or penalise the shortlist) before asking for the next — so a win is
   attributable to reasoning, not batch construction.
6. **Stronger surrogate, not stronger agent.** One-hot encodings contain zero chemistry —
   every ligand is equidistant from every other, so the GP *cannot* generalise across
   reagents early on. Both datasets ship DFT descriptors (density functional theory:
   quantum-chemistry-computed properties per reagent, e.g. HOMO/LUMO energies, partial
   charges, dipole moments — so chemically similar reagents get similar feature vectors;
   Ahneman et al. built their own model on exactly these). Rerun classic BO on those
   instead of one-hots: if the cold-start gap closes, part of the agent's edge was weak
   baseline featurization rather than reasoning.

Extensions:

7. **A more capable agent model.** The headline uses Gemini 2.5 Flash; our surrogate-coupled
   IMP@1 (41.9) trails the paper's Reasoning-BO (60.1), and cold start is exactly where
   model quality should matter. A stronger reasoning model (backend is already pluggable)
   would show whether the edge *scales* with capability. The mid-trajectory tie is less
   likely to move: there the surrogate, not the agent, carries the signal.
8. **A hybrid policy.** The findings suggest the practical recipe directly: agent for the
   first rounds (cold start), classic BO mid-game, agent again if progress stalls (trap
   escape). Test it as a fifth method.
9. **A genuinely continuous / higher-dimensional** reaction space, where a GP-from-scratch
   is weakest and the chemistry prior should matter most — the regime the cold-start result
   already hints at.
10. **Cost accounting.** Report API cost and wall-clock next to the yield numbers;
    "+9pp final best for cents of API calls" is the decision-relevant form of the result.
