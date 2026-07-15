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
(`results/decisions_*.jsonl`) with per-round spend, picks, values, request and
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
   below reports what an LLM's inferred trust adds.

Seed count matters here: at 4 seeds the ρ-dependence of `two_stage` disappears
into per-seed noise (the final value of a 5-measurement campaign has a spread
of tens of points); 16 seeds resolve the monotone trend.

The offline runs also confirm the mechanism the agent must show: the heuristic
simulates several times more often when the simulator is good than when it is
junk (14 vs 3 simulations per campaign on the synthetic pool at ρ=0.9 vs 0.2),
purely from the LF/HF error evidence in `recall`; the direction is pinned by a
test.

## The live agent (Gemini 2.5 Flash)

Two cells of the arylation sweep, ratio 10, 4 seeds, all calls cached:

| final, % of optimum | ρ=0.9 | ρ=0.3 |
|---|---|---|
| `two_stage` (16 seeds) | 90.8 | 88.3 |
| `agentic_mf`, heuristic (16 seeds) | 82.4 | 74.5 |
| `agentic_mf`, Gemini (4 seeds) | 74.1 ± 28.3 | 71.2 ± 10.7 |

The mechanism the study asks about is present, and the decision-log audit
makes it measurable. Between ρ=0.9 and ρ=0.3 the agent shifts its behaviour in
the right direction on every axis: 65% vs 34% of its measurements are
simulated first, the simulator error it observed is 11.4 vs 22.8, and
confirmed picks out-yield direct picks only where the simulator is good
(58.6 vs 41.5 at ρ=0.9; 33.5 vs 37.0 at ρ=0.3). The rationales state the
inference explicitly — at ρ=0.3 the agent calls the simulator unreliable and
switches to direct measurement.

The outcome does not beat the fixed strategies. Two behaviours explain most of
the gap. The agent runs one or two very long rounds instead of many small ones
(mean 8.8 requests per round at ρ=0.9; one round hit the request limit and was
logged as a fallback), and it stops voluntarily with budget left in 3 of 4
campaigns at ρ=0.9 — sometimes on a reasoned but premature "EI is low"
argument, which the ±28.3 spread reflects. Inferring trust works; spending the
budget well does not follow from it, and 4 seeds cannot rank the agent against
the baselines. A conclusion either way needs more seeds, a stricter stop
criterion in the prompt, and a higher request limit.

## What to trust

The LF oracle is constructed, ρ is pool-wide (a simulator can be locally much
worse), and the heuristic agent is a stand-in for the LLM, not a claim about
LLM behaviour. Budgets, costs, and seeds are explicit in every summary JSON;
curves count measurements only; and the agent's spending is auditable per round
in the decision log. The prompt and instructions are pinned by a golden test
because the replay cache is keyed on them.

## Open threads

* The full Gemini sweep at more seeds, with a stricter stop criterion and a
  higher per-round request limit (the two behaviours the first live cells
  flagged).
* Batched measuring for `mf_bo` and the agent (local penalization, as study 1).
* Cross-campaign memory: a later campaign on a *new* pool that shares the same
  simulator starts with the trust inferred earlier, instead of re-learning it.
  Only knowledge about the simulator transfers — candidate values do not, and
  seeds remain independent replications of the whole two-campaign sequence.
