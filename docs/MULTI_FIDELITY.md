# Multi-fidelity — when does a cheap simulator pay off?

Discovery campaigns usually have two oracles: a cheap simulation and an
expensive measurement. Each step the decision is not only which candidate to
try next but at which fidelity, and the simulator's reliability is not known in
advance. This study benchmarks that decision on a pool task with two oracles
and a budget in cost units instead of evaluation counts. The methods range from
ignoring the simulator, to trusting it blindly, to estimating its reliability
during the campaign.

Next to the research question, the study serves a second purpose: developing a
fuller agentic setup than study 1's. The agent runs on PydanticAI with typed
tools, tool-argument validation as retries, structured per-round decisions,
fallback handling around the LLM call, a replay cache, and an audited decision
log — and the same loop is driven deterministically in CI.
[TESTING.md](TESTING.md) describes the control layers around it.

## Terminology

* **Oracle** — anything that returns a value for a candidate when asked.
* **Fidelity** — how close an oracle is to the truth: low fidelity is the cheap
  simulation, high fidelity the expensive measurement.
* **Simulate (LF)** — query the cheap oracle, a corrupted version of the truth;
  costs 1.
* **Measure (HF)** — query the expensive oracle, the true value; costs `C`.
* **ρ (rho)** — how well the simulator correlates with the truth (1 = perfect,
  0 = uninformative). Unknown to the methods; the experiment's main knob.
* **Cost ratio** — how much more a measurement costs than a simulation (`C`).
* **Budget** — the total spend allowed, in cost units; simulations and
  measurements are paid from the same budget.
* **Campaign** — one full budgeted run on one pool: a few random initial
  measurements, then one method spends the rest until the budget is dry or it
  stops.
* **Fidelity mix** — the fraction of spend going to measurements over time.
* **Screen → confirm** — simulate broadly first, then measure the best-looking
  simulated candidates.

## Setup

**Two oracles, one ledger.** Every candidate can be *simulated* at cost 1 or
*measured* at cost `C` (the cost ratio). The low-fidelity oracle is constructed
from the true objective: standardise, mix with an independent random landscape,
rescale — so the LF/HF correlation is an explicit knob **ρ**
([oracles.py](../src/multi_fidelity/oracles.py)). The corruption is drawn once
per problem and the simulator is deterministic: re-querying a candidate returns
the same number and buys nothing, like a systematically wrong model and unlike
mere noise. No real cheap simulator exists for these datasets, so constructing
one is unavoidable — standard in multi-fidelity benchmarks. With ρ explicit,
the effect of simulator quality is measured rather than assumed.

**Only measurements count.** All spending goes through a cost ledger, and the
reported curve is *best measured value per unit cost*; a great simulated score
is a claim, not a result. The second reported curve is the **fidelity mix** —
the fraction of spend going to measurements over time — which makes a method's
trust in the simulator visible, not just its outcome.

## Methods

| method | uses the simulator | how |
|---|---|---|
| `random_hf` | no | measure random candidates; the floor |
| `classic_bo_hf` | no | GP + EI on measurements only; the single-fidelity ceiling |
| `two_stage` | blindly | spend half the budget screening at LF, measure the screen's top scorers |
| `mf_bo` | modelled | one GP over (features, fidelity flag); EI at HF, simulate-first while the LF value is uncertain |
| `agentic_mf` | inferred | a PydanticAI agent spends the budget through tools and judges the simulator from its own LF/HF pairs |

The agent ([agent.py](../src/multi_fidelity/agent.py)) gets five typed tools —
`recall` (campaign state, including simulator-vs-truth errors on shared
candidates), `shortlist` (surrogate-ranked candidates), `predict` (free
surrogate queries on candidates of its own choosing, so it can check its own
hypotheses), and the two oracles, which really spend budget — and a standing
instruction that the simulator's reliability is unknown and must be inferred.
Invalid calls (re-queries, overspends, bad ids) are bounced back as
`ModelRetry` and cost nothing; a round that fails outright (API error, retries
exhausted) is logged as a fallback and the campaign continues. Each round ends
in a structured decision (rationale, a scratchpad note carried forward,
optional stop); budget, not round count, ends the campaign.

Offline, the same agent loop is driven by a deterministic heuristic packaged as
a PydanticAI `FunctionModel` ([heuristic.py](../src/multi_fidelity/heuristic.py)):
trust the simulator until its mean absolute error on shared candidates exceeds
half the spread of the measurements, then degrade to direct measuring. CI
exercises the real loop — tools, validation, retries, structured output — with
no API. LLM legs run behind a replay cache
([cache.py](../src/multi_fidelity/cache.py)) keyed on the full serialized
request, so identical reruns are free and any prompt change is a cache miss by
construction.

## How to run

```bash
uv run run_multi_fidelity.py                          # offline single config (synthetic pool)
uv run run_multi_fidelity.py --sweep                  # rho x cost-ratio grid
uv run run_multi_fidelity.py --dataset arylation --sweep --seeds 16              # table below
uv run run_multi_fidelity.py --dataset buchwald --sweep --seeds 8 --subsample 1000  # contrast

export GEMINI_API_KEY=...                             # or via .env; needs --extra mf
uv run run_multi_fidelity.py --dataset arylation --agent gemini --seeds 3 --verbose
```

Single-config runs produce a two-panel plot (convergence per unit cost;
fidelity mix over time), a summary JSON, and — for the agent — a decision log
(`results/multi_fidelity/decisions_*.jsonl`) with per-round spend, picks, values, request and
retry counts, and rationales. `scripts/audit_mf_decisions.py` reports on a log
without API calls: the fidelity mix early vs late, the screen-confirm gain,
the simulator error the agent itself observed, and whether campaigns ended
budget-dry or by a voluntary stop. The test stack behind the agent is described
in [TESTING.md](TESTING.md).

## Results (offline heuristic agent, budget 200)

Final best measurement as % of the pool optimum. Arylation (the hard, deceptive
pool of study 1; 16 seeds):

| | ρ=0.9 | | | ρ=0.3 | | |
|---|---|---|---|---|---|---|
| **cost ratio** | 5 | 10 | 20 | 5 | 10 | 20 |
| `random_hf` | 90.3 | 81.8 | 56.0 | 90.3 | 81.8 | 56.0 |
| `classic_bo_hf` | 88.3 | 74.0 | 58.0 | 88.3 | 74.0 | 58.0 |
| `two_stage` | 92.3 | 90.8 | 88.7 | 90.2 | 88.3 | 81.4 |
| `mf_bo` | 92.7 | 82.9 | 67.7 | 87.0 | 80.8 | 61.5 |
| `agentic_mf` (heuristic) | 93.2 | 82.4 | 58.1 | 90.3 | 74.5 | 58.6 |

Buchwald (the easy pool, GP-friendly; 8 seeds, subsample 1000):

| | ρ=0.9 | | | ρ=0.3 | | |
|---|---|---|---|---|---|---|
| **cost ratio** | 5 | 10 | 20 | 5 | 10 | 20 |
| `random_hf` | 91.2 | 87.1 | 75.8 | 91.2 | 87.1 | 75.8 |
| `classic_bo_hf` | 97.2 | 91.6 | 86.3 | 97.2 | 91.6 | 86.3 |
| `two_stage` | 95.0 | 93.7 | 90.7 | 89.3 | 83.9 | 76.2 |
| `mf_bo` | 97.0 | 94.8 | 87.2 | 94.7 | 88.5 | 80.7 |
| `agentic_mf` (heuristic) | 96.8 | 94.6 | 82.0 | 97.0 | 91.0 | 79.9 |

Four readings:

1. **The cost ratio decides whether the simulator matters at all.** At ratio 5
   the budget buys ~40 measurements and every method lands above 88% on both
   pools. At ratio 20 the budget buys ~10 measurements and the gap opens.
2. **On the hard pool, even a bad screen beats no screen.** Arylation has ~32%
   dead conditions, so any weakly correlated screen steers the few measurements
   away from them: at ratio 20, `two_stage` scores 81–89% across the whole ρ
   range while the single-fidelity methods sit at 56–58%. Simulator quality
   still shows — `two_stage` drops monotonically from 88.7 (ρ=0.9) to 81.4
   (ρ=0.3) — but breadth carries most of the value.
3. **On the easy pool, blind trust loses to no trust.** Classic BO is already strong
   on Buchwald (86–97%), and at ρ=0.3 `two_stage` falls *below* it at every
   cost ratio (e.g. 76.2 vs 86.3 at ratio 20): budget spent confirming a junk
   screen is budget taken from a method that was winning without the simulator.
   `mf_bo`, which learns the fidelity relation, degrades more gracefully and
   stays above `two_stage` at low ρ.
4. **The adaptive methods sit between the extremes offline.** Neither `mf_bo`
   nor the heuristic agent dominates both regimes; the live-agent section
   below reports what an LLM's inferred trust adds — on this task, less than
   the heuristic's mechanical rule.

Seed count matters here: at 4 seeds the ρ-dependence of `two_stage` disappears
into per-seed noise (the final value of a 5-measurement campaign has a spread
of tens of points); 16 seeds resolve the monotone trend.

The offline runs also confirm the mechanism the agent must show: the heuristic
simulates several times more often when the simulator is good than when it is
junk (14 vs 3 simulations per campaign on the synthetic pool at ρ=0.9 vs 0.2),
purely from the LF/HF error evidence in `recall`; the direction is pinned by a
test.

## The live agent (Gemini 2.5 Flash)

The full sweep with an LLM in the loop, 8 seeds per cell, all calls cached
(`uv run run_multi_fidelity.py --dataset arylation --sweep --seeds 8
--methods agentic_mf --agent gemini`). Final best measurement, % of the pool
optimum, next to the offline methods from the table above:

| | ρ=0.9 | | | ρ=0.6 | | | ρ=0.3 | | |
|---|---|---|---|---|---|---|---|---|---|
| **cost ratio** | 5 | 10 | 20 | 5 | 10 | 20 | 5 | 10 | 20 |
| `two_stage` (16 seeds) | 92.3 | 90.8 | 88.7 | 91.9 | 89.9 | 86.7 | 90.2 | 88.3 | 81.4 |
| `mf_bo` (16 seeds) | 92.7 | 82.9 | 67.7 | 92.2 | 78.9 | 62.4 | 87.0 | 80.8 | 61.5 |
| `agentic_mf`, heuristic (16 seeds) | 93.2 | 82.4 | 58.1 | 96.2 | 84.3 | 67.5 | 90.3 | 74.5 | 58.6 |
| `agentic_mf`, Gemini (8 seeds) | 91.1 | 81.9 | 56.8 | 83.5 | 66.8 | 58.4 | 81.4 | 69.2 | 50.1 |

**The LLM agent loses to every fixed strategy**, in every cell, and to the
deterministic heuristic in seven of nine. It tracks `random_hf` (90.3 / 81.8 /
56.0 by cost ratio) rather than the multi-fidelity methods: the budget is
spent, but not on the right fidelity at the right time.

The decision-log audit locates the failure precisely. Loop mechanics are
healthy — across all nine cells there are **zero fallback rounds, zero
premature stops** (every campaign runs until the budget cannot buy another
measurement), 4–5 requests per round, and ~4 rounds per campaign. The problem
is that **the fidelity mix does not respond to ρ**: the agent runs 10–20
simulations per campaign and spends 86–94% of its budget on measurements at
*every* simulator quality. The evidence is in front of it — the simulator
error it observed is 9.4 at ρ=0.9, 18.0 at ρ=0.6, 23.8 at ρ=0.3, and its
rationales name the simulator unreliable at low ρ — but the spend barely
moves. Screening only pays where the simulator is good (measurements it
simulated first average 53.2 vs 41.3 for direct ones at ρ=0.9; 36.7 vs 44.6 at
ρ=0.6), so a fixed mix is wrong at both ends: too little screening at ρ=0.9,
too much at ρ=0.3.

Part of that is a **tool-design problem, not a reasoning problem**. The winning
behaviour on this task is a wide cheap screen — `two_stage` simulates 40–100
candidates in one step — but the agent's only way to screen is
`simulate(candidate_ids)` with the ids written out one by one, and the ids
have to come from `shortlist`, which returns a top-k of about eight. Screening
broadly therefore costs the agent long argument lists and several rounds,
while measuring is a one-id call. The tools make the strategy that wins
expensive to express, and the observed mix of ~15 simulations per campaign is
about what a top-k-sized shortlist affords. The next version of the agent needs
screening primitives at the same granularity as its decisions: something like
`simulate_top(n)` or `simulate_random(n)` over the whole pool, plus a
`shortlist` whose size the agent chooses.

An earlier 4-seed pair of cells, run before the prompt was tightened, appeared
to show the mix tracking ρ (65% vs 34% of measurements simulated first).
That signal did not reproduce here at 8 seeds. Two explanations are open: it
was noise at 4 seeds, or the stricter "spend the whole budget" instruction
traded trust-adaptation for spend. Distinguishing them needs the old prompt
re-run at 8 seeds; the current numbers do not settle it.

What the study can say: inferring simulator reliability from evidence is
something the agent demonstrably does in its reasoning, and acting on that
inference in its spending is something it does not — with the caveat that its
tools currently price the adaptive strategy out. On this task a five-line
heuristic that mechanically compares simulator error against the spread of the
measurements converts the same evidence into better decisions.

## What to trust

The LF oracle is constructed, ρ is pool-wide (a simulator can be locally much
worse), and the heuristic agent is a stand-in for the LLM, not a claim about
LLM behaviour. Budgets, costs, and seeds are explicit in every summary JSON;
curves count measurements only; and the agent's spending is auditable per round
in the decision log. The prompt and instructions are pinned by a golden test
because the replay cache is keyed on them.

## Open threads

* **Give the agent screening tools that match the strategy.** Its spend cannot
  track ρ while a broad screen has to be spelled out id by id from an
  eight-item shortlist. Add pool-wide primitives (`simulate_top(n)`,
  `simulate_random(n)`, an agent-chosen shortlist size) and rerun the sweep:
  this is the first thing to fix, because it is the one that currently prices
  the adaptive strategy out.
* Then separate the two remaining explanations for the flat mix: the prompt
  rewards spending over adapting (test by rerunning the pre-tightening prompt
  at 8 seeds), or the model states the inference without acting on it — which
  is what remains if better tools do not change the mix, and matches study 1's
  finding that the agent followed max-EI 93% of the time whatever its rationale
  said.
* Batched measuring for `mf_bo` and the agent (local penalization, as study 1).
* Cross-campaign memory: a later campaign on a *new* pool that shares the same
  simulator starts with the trust inferred earlier, instead of re-learning it.
  Only knowledge about the simulator transfers — candidate values do not, and
  seeds remain independent replications of the whole two-campaign sequence.
