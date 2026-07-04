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

## 6. First credibility check: auditing the agent's rationale

Every decision now lands in `results/decisions_<tag>.jsonl` (shortlist + ground truth next
to the agent's stated `strategy`/`rationale`), and because the decision cache replays the
headline run for free, we could audit the Gemini arylation run retroactively — zero API
calls (`scripts/audit_decisions.py`; 186 of its 200 decisions replayed from cache, the 14
cache misses are flagged as fallbacks and excluded).

What the audit says about the surrogate-coupled agent:

- **It follows max-EI 93% of the time** (80/86 decisions). The "agent" is mostly
  rubber-stamping the acquisition function — direct evidence for the honest label of §"next
  steps" 11: this is LLM-*guided* BO, not an autonomous agent.
- **The 6 overrules paid off**: picking against max-EI gained **+9.9 yield points** on
  average vs the EI candidate (won 4/6). The agent's measurable added value is concentrated
  in a handful of decisive moments — the same shape as the §4 conclusion (cold start + trap
  escape, on par elsewhere).
- **The stated strategy is connected to behaviour, not decoration**: 96% of "exploit"
  decisions picked from the shortlist's top-3 by predicted mean, and "exploit" rounds
  out-yielded "explore" rounds (56 vs 3 — though only 2 explore rounds exist to compare).
- **Without the surrogate, the prior is directly visible**: the top pick sits at the
  **79–82nd percentile** of its (random) shortlist by true yield, where a blind pick sits
  at 50. Late-run "exploit" decisions (mean round 8.5) score higher than early "balance"
  ones (52 vs 37 yield) — in-context learning on top of the prior.

**Lesson:** a rationale audit is cheap insurance against confabulation, and it sharpened
the story: the surrogate does the routine work; the model earns its keep in rare,
high-value overrides. Whether the 79th-percentile prior is *chemistry* or *memorisation*
is exactly what the leakage probes (below) must decide.

## 7. Running the credibility checks: the edge is reasoning, not recall

We ran all three checks against Gemini 2.5 Flash on the arylation task (zero-shot probe:
40 trials; name ablation and permuted yields: the full paper protocol, 10 seeds each —
~640 flash calls in total, cents not dollars). Together they *change the story* of §4.

**Zero-shot probe** (`scripts/leakage_probe.py`) — one pick per seeded random shortlist,
no measurements shown, scored against the ground-truth pool:

| zero-shot, 40 trials | pool percentile | pick yield | picked shortlist best |
| --- | --- | --- | --- |
| blind pick (baseline) | 54% | 19.4 | 12% |
| Gemini, named reagents | **67%** | **32.3** | 25% |
| Gemini, anonymised | 57% | 25.8 | 20% |

A real but modest zero-shot prior that all but evaporates when reagent names are
withheld — so *whatever* the model knows cold, it accesses through the names.

**Name ablation in the loop** (`--anonymize`, full protocol): cold start barely moves —
no-surrogate IMP@1 goes 53.0 → 49.4 with a ±30 per-seed spread (classic BO: 36.2). The
in-loop cold-start edge therefore does **not** come from named chemistry knowledge. The
logged rationales show the actual mechanism: reuse components of the best of the three
init observations and vary one factor at a time — few-shot *combinatorial* reasoning over
the observed data, which EI over one-hot encodings cannot do at round one. Names do matter
mid-game (no-surrogate IMP@3 drops 47.3 → 28.8; with surrogate 67.0 → 59.2), consistent
with chemistry knowledge helping to avoid dead regions once there is history to interpret.
The final is untouched (96% → 95%).

**Permuted yields** (`--permute-yields`, full protocol): with chemistry decoupled from
reward, every agent edge collapses to random-or-worse — cold start 53.0 → 32.3, final 96%
→ 79%, *below* random's 87% (structured reasoning actively hurts in a structureless
world). The leakage indicator is clean: the picks' percentile under the *original* yields
(never shown to the model) is 56–60% vs the ~50% of a blind pick — nowhere near the
79–82% the same agent scores when the yields are real. If the model were recalling the
dataset, permuting the yields could not have erased that signal.

**Verdict:** no evidence that the headline numbers are dataset recall. But the checks
sharpened §4 in an unexpected way: the agent's cold-start edge is less "knows chemistry"
than "reasons combinatorially from three data points" — a mechanism the zero-shot probe
alone would have misattributed to chemistry knowledge, and one that should transfer to
search spaces the model has never seen. That is a *more* encouraging result than the one
it replaces, and we only own it because the counterfactual runs were cheap to ask for.

## Caveats / limitations

- Absolute parity with the paper is not achievable: their search space is under-specified
  (likely continuous concentration/temperature vs our discrete 1728-grid), and our LP batch
  is a stand-in for their qLogEI. Compare *patterns*, not decimals.
- The synthetic MOF objective is physically motivated but not real data.
- IMP@1 for the surrogate-coupled agent (41.9) is below the paper's Reasoning-BO (60.1); its
  first batch spreads wider than theirs. Round-one behaviour is not fully matched.

## What I'd do next

Credibility checks — **all three are done** (tooling ships in the repo; see the README's
credibility-checks section):

1. **Data-leakage check.** *(done — findings in §7.)* Zero-shot probe
   (`scripts/leakage_probe.py`) plus `run.py --permute-yields`: modest name-carried
   zero-shot prior, everything collapses to random under permuted yields, leakage
   indicator clean — no evidence of dataset recall.
2. **Name-ablation prompt.** *(done — findings in §7.)* `run.py --anonymize`: the in-loop
   cold-start edge survives without names (it is few-shot combinatorial reasoning over the
   init observations); names matter zero-shot and mid-game.
3. **Rationale audit.** *(done — findings in §6.)* Decisions are logged with ground truth
   (`results/decisions_<tag>.jsonl`) and `scripts/audit_decisions.py` checks that stated
   strategies are connected to reality. Remaining: spot-check the *chemical claims* in the
   rationales by hand (`--show`) against the full pool.

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
11. **Make the agent actually agentic.** Today the "agent" is one stateless, structured LLM
    call per round — the harness does the interesting work (fits the GP, computes EI, builds
    the shortlist) and the LLM only ranks ~8 pre-digested options. "LLM-guided BO" is the
    honest label. The agentic upgrades, in rough order of expected value:
    - **Tool use:** let the model query the surrogate itself ("predict these 5 points"),
      request more candidates, or compute — instead of receiving a fixed shortlist.
    - **Control over the loop:** let it choose batch size q, stop early, or tune the
      explore/exploit mix of the shortlist.
    - **Persistent memory:** a scratchpad carried across rounds ("aryl halide X consistently
      underperforms") instead of re-reading raw history each round.
    - **Multi-step deliberation:** hypothesise → verify against the surrogate → pick,
      rather than one shot.
