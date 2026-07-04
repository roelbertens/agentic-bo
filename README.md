# Agentic Bayesian Optimisation for Scientific Discovery

A compact, reproducible study of an **agentic Bayesian-optimisation (BO) loop**: an LLM
agent that decides which experiment to run next, benchmarked against classic BO across a
synthetic materials task and two **real reaction-optimisation datasets**. On the hard one it
reproduces a published LLM-BO result — see the [headline result](#headline-result--direct-arylation-hard-task-reasoning-bo-protocol)
and the honest write-up in [LEARNINGS.md](LEARNINGS.md).

Five policies compete on the same task, using as few (simulated) experiments as possible.
For a step-by-step, side-by-side walkthrough of *how each one decides* on one identical
scenario (with real numbers and a visual), see **[docs/METHODS.md](docs/METHODS.md)**.

| Policy | What it does |
| --- | --- |
| **Random search** | Naive floor — pick the next candidate at random. |
| **Classic BO** *(baseline)* | Gaussian-process surrogate + Expected Improvement; batches via local penalization (a fair stand-in for qLogEI). |
| **Agentic BO** | An **agent** (Gemini / Claude, or a heuristic stand-in) chooses the next experiment from a surrogate-informed shortlist, reasoning about explore/exploit each round. |
| **Agentic BO – no surrogate** *(ablation)* | The same agent, but **without** the GP's mean/std/EI — isolates how much the surrogate contributes vs. the agent's own knowledge. |

The point is not a new method. It is an honest, small-scale answer to a practical question —
*does putting an LLM in the BO loop actually help, and when?* — with the evaluation hygiene
(fair baseline, ablation, multi-seed spread) needed to trust the answer.

## Three datasets — increasing in how much an agent can help

**`mof`** *(synthetic, default)* — CO₂ working capacity of metal-organic frameworks as a
smooth function of five physical descriptors. Classic GP+EI is already near-optimal on a
landscape this smooth, so an agent has little to add. Useful as a sanity check and a floor.

**`buchwald`** *(real, easy)* — the Buchwald-Hartwig HTE dataset of Ahneman et al.
(*Science* 2018; often called the Dreher-Doyle dataset): a complete grid of **3,955**
Pd-catalysed C-N cross-couplings
(4 ligands × 3 bases × 22 additives × 15 aryl halides) with measured yields. Discrete and
built of **named building blocks** (XPhos, P2Et, …) — but good conditions are *dense*
(~7% of the pool yields ≥80%), so even random reaches ~85% of the optimum and every method
saturates. A useful "is the agent at least competitive?" check.

**`arylation`** *(real, hard)* — the Shields et al. direct C-H arylation dataset (*Nature*
2021): a full-factorial pool of **1,728** reactions (12 ligands × 4 bases × 4 solvents × 3
concentrations × 3 temperatures). Deceptive: **~32% of conditions give ~0% yield** and the
median is ~8%, so a GP starts blind and cold-start prior knowledge (avoid dead
ligand/base/solvent combinations) matters most. This is the regime where LLM-BO has been
reported to clearly beat classic BO — e.g. Ramos et al. and Reasoning-BO (see
[References](#references)).

Because both datasets are finite with known ground truth, the pool optimum is exact and
"% of optimum" needs no API to compute.

### Three things that make the comparison *fair*

Good reactions are dense enough that best-of-N saturates — even random reaches ~89% of the
optimum, so the *final* metric barely separates methods. Getting a trustworthy comparison
took three corrections (the full story is in [LEARNINGS.md](LEARNINGS.md)):

- **IMP@k, not final.** With `--batch-size > 1` the summary reports **IMP@k** = the per-round
  *proposal quality* (best yield in round k's batch, non-cumulative), the metric Reasoning-BO
  (2025) uses. That is where cold-start prior knowledge shows up; the final saturates.
- **A fair batch baseline.** Classic BO batches via **local penalization** (spread picks out),
  not greedy top-q EI (near-duplicate picks that get stuck). The weak version let *random* beat
  classic BO — a baseline artifact, not a real result.
- **Multiple seeds.** 10 seeds; the gaps sit inside ±7–13, so single-run "leads" are treated
  as noise until they survive the spread.

(`--per-substrate` is also available for Buchwald: fix the aryl halide and optimise the 264
ligand/base/additive combinations, averaged over substrates.)

## Quick start

Uses [uv](https://docs.astral.sh/uv/); no uv? `pip install -e ".[gemini]"` works too
(then drop the `uv run` prefix).

```bash
uv sync                             # offline runs; add --extra gemini for the LLM agent

# Fast, fully offline — heuristic agent stands in for the LLM (no API key):
uv run run.py                       # synthetic MOF task
uv run scripts/get_data.py          # fetch the reaction datasets (~2 MB, once)
uv run run.py --dataset buchwald    # real reaction task, heuristic baselines

# The real agentic loop on the reaction task, with Gemini:
export GEMINI_API_KEY=...
uv run run.py --dataset buchwald --agent gemini --seeds 3 --budget 20 --verbose

# The fair "can the agent beat classic BO?" test — per-substrate + cold start:
uv run run.py --dataset buchwald --per-substrate --substrates 4 \
              --n-init 3 --budget 12 --agent gemini --seeds 3 --verbose

# ...or with Claude (--extra claude, --agent claude, ANTHROPIC_API_KEY).

# Reproducing Reasoning-BO's Direct Arylation protocol (batch 3, IMP@k metric):
uv run run.py --dataset arylation --agent gemini \
              --n-init 3 --budget 30 --batch-size 3 --seeds 10 --verbose
```

**Comparing to the literature.** With `--batch-size > 1` the summary reports **IMP@k**
(the per-round *proposal quality* — the best yield among round k's batch, non-cumulative),
matching the metric in Reasoning-BO (2025). As a validation, our random search reproduces
their random baseline closely (IMP@1 ≈ 30 vs 29, including the non-monotonic dip). Note the
final best-so-far converges high for every method (their Log-AUC column agrees) — the
signal lives in the early IMP@k, not the final. Exact numeric parity is limited by their
under-specified search space (likely continuous concentration/temperature vs our discrete
1728-grid) and their qLogEI vs our local-penalization batch, so compare *patterns*, not decimals.

Outputs land in `results/`: `convergence_<dataset>_<agent>.png` and a `summary_*.json`.
Run the tests with `uv run pytest -q`.

## Headline result — direct arylation (hard task), Reasoning-BO protocol

Gemini 2.5 Flash agent, paper protocol (n_init 3, batch 3, 30 experiments, 10 seeds), with
`classic_bo` on a fair local-penalization batch. **IMP@k** = per-round proposal quality
(matches Reasoning-BO); `final` = best-so-far as % of the pool optimum:

![convergence](results/convergence_arylation_gemini.png)

| method | IMP@1 | IMP@3 | IMP@5 | final |
| --- | --- | --- | --- | --- |
| random | 30.4 | 25.3 | 48.7 | 89% |
| classic BO (GP + EI, LP batch) | 36.2 | 61.6 | 70.0 | 87% |
| **agentic BO** (Gemini + surrogate) | 41.9 | 67.0 | **71.7** | **96%** |
| agentic BO, no surrogate | **53.0** | 47.3 | 47.2 | 92% |
| *paper — Vanilla BO* | 43.6 | 45.2 | 55.9 | — |
| *paper — Reasoning-BO* | 60.1 | 66.6 | 71.2 | — |

Read honestly (full arc in [LEARNINGS.md](LEARNINGS.md)):

- **Cold start (IMP@1) — real agent win.** Chemistry-only (no surrogate) scores 53 vs BO's 36:
  it picks good ligand/base/solvent conditions from round one, which a GP-from-scratch cannot.
- **Final — real agent win.** 96% vs 87%: classic BO over-exploits and caps out on this
  deceptive landscape (32% of conditions are dead); the agent escapes the trap.
- **Mid-trajectory (IMP@5) — a tie** against the *fair* baseline (71.7 vs 70.0). Against a weak
  greedy-EI batch the agent looked far ahead; most of that gap was the baseline. Scrutinising
  the random result exposed it.

On the **easy** `buchwald` task the opposite holds — the surrogate *hurts* the LLM, because
greedy EI is already near-optimal and the agent only adds noise. **Whether an LLM helps BO is
dataset-dependent**; benchmark on the regime that matches your problem.

## Credibility checks — is the agent's edge real?

Both datasets are public and plausibly in LLM training data, and the agent returns a
free-text rationale that could be confabulated. The repo ships three tools to attack its
own headline claim (next steps 1–3 in [LEARNINGS.md](LEARNINGS.md)):

**Rationale audit** *(no API calls)* — every agentic run writes
`results/decisions_<tag>.jsonl`: per decision the full shortlist with ground truth next to
the agent's stated `strategy`/`rationale`. The audit checks the reasoning against reality —
do "exploit" rounds out-yield "explore" rounds, do picks match the stated strategy, and
when the agent overrules max-EI, does that pay off?

```bash
uv run scripts/audit_decisions.py results/decisions_arylation_gemini.jsonl --show 5
```

**Zero-shot leakage probe** — asks the model for its best pick with *zero measurements
shown*, over many seeded random shortlists, and scores the picks against the ground-truth
pool. Near the blind baseline = no usable prior; well above = prior knowledge (chemistry
*or* memorisation — the next two flags separate those):

```bash
uv run scripts/leakage_probe.py --dataset arylation --agent gemini --trials 20
uv run scripts/leakage_probe.py --dataset arylation --agent gemini --trials 20 --anonymize
```

**Name ablation & permuted yields** — two `run.py` flags that rerun the benchmark under
counterfactual conditions:

- `--anonymize` withholds reagent names/SMILES (opaque ids only). If the cold-start edge
  disappears, it came from *named* chemistry knowledge — the claimed mechanism.
- `--permute-yields` shuffles the yields across candidates (seeded), so chemistry no longer
  maps to reward. The agent should fall to random; the decision log keeps both the observed
  and the original yields, and the audit's leakage indicator flags a model that keeps
  chasing the *original* optimum it was never shown (memorisation, not reasoning).

Outputs get their own tags (`..._anon_...`, `..._permuted_...`), so they never overwrite
the headline results.

**What they showed** (Gemini 2.5 Flash, arylation — full numbers in
[LEARNINGS.md §6–7](LEARNINGS.md)): the rationale audit found the agent follows max-EI 93%
of the time but its rare overrules gain +9.9 yield points; the zero-shot prior is modest
and carried by reagent names; the in-loop cold-start edge survives anonymisation (it is
few-shot combinatorial reasoning over the init observations, not named chemistry); and
under permuted yields everything collapses to random with a clean leakage indicator — **no
evidence the headline result is dataset recall**.

## Structured evals with Langfuse (optional)

The decision log is the source of truth; [Langfuse](https://langfuse.com) (MIT, self-hostable)
adds the structured layer on top: a trace UI to drill into any single decision, first-class
scores, and session-level A/B comparison across policies and seeds. The mapping is
**session = run tag, trace = policy × seed campaign, span = decision round**, with the
audit's own metrics attached as scores (`pick_percentile`, `follows_max_ei`,
`overrule_gain_y`, `batch_best_y`, `fallback`, `cache_hit`) — computed by the same code
(`agentic_bo/tracing.py:derived_scores`), so the UI and `audit_decisions.py` can never
disagree.

```bash
uv sync --extra eval
export LANGFUSE_PUBLIC_KEY=... LANGFUSE_SECRET_KEY=...
export LANGFUSE_HOST=http://localhost:3000    # self-hosted; omit for Langfuse cloud

# Live: trace a run as it happens (adds per-decision LLM latency):
uv run run.py --dataset arylation --agent gemini --batch-size 3 --langfuse

# Backfill: push an existing decision log — cached runs included, zero API calls:
uv run scripts/push_to_langfuse.py results/decisions_arylation_gemini.jsonl
```

To self-host: `git clone https://github.com/langfuse/langfuse && cd langfuse &&
docker compose up`, then create a project at `http://localhost:3000` and copy its keys.
Tracing is off by default, needs no dependencies unless enabled, and fails soft — a
missing package, key, or server prints one warning and never touches the run.

## Making the agent *actually* agentic (tool use, memory, deliberation)

The `agentic_bo` policy above is honestly **LLM-guided BO**: one stateless call per round
that ranks ~8 pre-digested options while the harness does the real work. The `agentic_tools`
policy ([agentic.py](src/agentic_bo/agentic.py)) is the genuinely-agentic version — each
round the model *drives* the loop through Gemini function-calling, in two phases that can
use **different models**:

- **Investigate** (worker model) — a tool loop over the real BO state: `predict` the
  surrogate on any candidates it names, `search_candidates` across the whole pool itself
  (by EI / mean / uncertainty / reagent match / random) instead of a fixed shortlist, and
  `recall` a scratchpad carried across rounds. Ends with `report`.
- **Deliberate** (reasoner model) — given the report, the measured history and its memory,
  optionally `predict` a few candidates to verify a hypothesis, then `submit` the batch with
  a `strategy`/`rationale` and a `note` appended to memory.

The research question this sets up: **does a stronger reasoner at the decision step (with a
cheap worker exploring) beat the single-call agent?** Route the two steps independently:

```bash
# Tool-agency alone (same model both phases) vs a stronger reasoner at the decision step:
uv run run.py --dataset arylation --agent gemini --methods classic_bo agentic_bo agentic_tools \
              --worker-model gemini-2.5-flash --reasoner-model gemini-2.5-flash \
              --n-init 3 --budget 30 --batch-size 3 --seeds 10
uv run run.py --dataset arylation --agent gemini --methods agentic_tools \
              --worker-model gemini-2.5-flash --reasoner-model gemini-2.5-pro \
              --n-init 3 --budget 30 --batch-size 3 --seeds 10
```

The batch size `q` and total budget stay **fixed** for a fair head-to-head with
`agentic_bo` (the §3 lesson again); the agent may *state* a preferred batch size or a
"stop now" — logged for analysis, not acted on. Runs are tagged by routing
(`..._tools_flash-flash_...`, `..._tools_flash-pro_...`) so configs never clobber. Offline,
`--agent heuristic` uses a deterministic tool agent (the CI path and knowledge-free floor).

**Does it help?** (arylation, 10 seeds, full write-up in [LEARNINGS.md §8](LEARNINGS.md)):

| method | IMP@1 | IMP@3 | IMP@5 | final |
| --- | --- | --- | --- | --- |
| agentic BO — single call (flash) | 41.9 | 66.9 | 71.4 | **95%** |
| agentic tools — flash + flash | 55.0 | 54.5 | **73.6** | 86% |
| **agentic tools — flash + Pro** | **57.8** | 66.7 | 68.5 | 90% |

Tool-agency buys a large, robust **cold-start** gain (IMP@1 42 → 55–58, beating even the
knowledge-only agent), and routing a **stronger reasoner to the decision step** (flash→Pro)
repairs the mid/late convergence that pure tool-agency softens. But it is *not* a uniform
win — the simple single call still converges best on the final metric. The honest read is a
hybrid: agentic tool loop for cold start, plain agent / classic BO for late convergence.

## How the agentic loop works

Each round the agentic policy:

1. Fits the GP surrogate on everything measured so far.
2. Builds a **shortlist** — mostly top-Expected-Improvement candidates plus a couple of
   high-uncertainty ones, so exploration is always an option.
3. Hands the agent the run history + each candidate (as chemistry, with a reagent legend)
   and, when available, the surrogate's mean ± std and EI.
4. The LLM returns a **structured decision** (`picks`, `strategy`, `rationale`) via
   the provider's JSON-schema / structured-output mode; the measured value is fed back and
   the loop repeats.

Agent backends are pluggable (`src/agentic_bo/agent.py`), all behind one `select_batch`
interface: `HeuristicAgent` (deterministic, runs anywhere), `GeminiAgent` (`google-genai`,
default `gemini-2.5-flash`), `AnthropicAgent` (`anthropic`, default `claude-opus-4-8`).
Set `--model` to override; each round the agent returns a ranked batch of `--batch-size` picks.

## Layout

```
run.py                     CLI: run the benchmark, write plot + summary + decision log
scripts/get_data.py        download the reaction datasets (Buchwald + arylation)
scripts/audit_decisions.py rationale audit: check the agent's reasoning against ground truth
scripts/leakage_probe.py   zero-shot data-leakage probe (with/without reagent names)
scripts/push_to_langfuse.py backfill a decision log into Langfuse (traces + audit scores)
src/agentic_bo/
  data.py                  dataset-agnostic container + permute-yields leakage check
  objective.py             synthetic MOF pool + ground-truth objective
  reactions.py             real loaders: Buchwald-Hartwig + direct arylation (named reagents)
  surrogate.py             GP surrogate + Expected Improvement
  agent.py                 decision context + Heuristic / Gemini / Claude backends (single call)
  agentic.py               genuinely-agentic tool loop: env + tools, two-phase agent, memory
  policies.py              random, classic BO, agentic BO (+ ablation), agentic-tools
  experiment.py            multi-seed runs, convergence curves + per-round IMP@k
  plotting.py              convergence + regret figures
  cache.py                 persistent prompt->decision cache (free resume)
  tracing.py               optional Langfuse tracing (no-op unless --langfuse)
tests/                     end-to-end + credibility-tooling checks (offline)
.claude/skills/            repo skills: quality-gate (pre-commit) + experiment-hygiene
.github/workflows/ci.yml   CI: ruff + pytest on 3.10 and 3.13
LEARNINGS.md               what the experiments actually taught us
```

## Notes

- `--agent gemini/claude` makes one API call per round per seed (a round proposes
  `--batch-size` candidates). Decisions are cached to `.cache/`, so an interrupted run
  resumes for free and re-analysis (new baseline, new plot) costs no API calls.
- The reaction datasets are downloaded, not committed (see `.gitignore`); rerun
  `scripts/get_data.py` on a fresh clone.
- Development: `uv run pytest -q` (offline) and `uv run --group dev ruff check .` must both
  be clean; CI enforces them. The repo also ships two [Claude Code](https://claude.com/claude-code)
  skills in `.claude/skills/` — a pre-commit **quality-gate** (tests, lint, no secrets,
  docs in sync, prompt/cache stability) and **experiment-hygiene** (seeds, metrics,
  fallback checks, when results may be reported) — so agent-assisted changes are held to
  the same standards as manual ones.
- See **[LEARNINGS.md](LEARNINGS.md)** for the honest write-up: metric choice, dataset-
  dependence of the agent's value, reproducing Reasoning-BO, and baseline fairness.

## References

Datasets (downloaded by `scripts/get_data.py`, not redistributed here):

- Ahneman, D. T., Estrada, J. G., Wang, S., Dreher, S. D. & Doyle, A. G.
  *Predicting reaction performance in C-N cross-coupling using machine learning.*
  Science 360, 186-190 (2018). [doi:10.1126/science.aar5169](https://doi.org/10.1126/science.aar5169).
  File obtained from [rxn4chemistry/rxn_yields](https://github.com/rxn4chemistry/rxn_yields) (MIT).
- Shields, B. J. et al. *Bayesian reaction optimization as a tool for chemical synthesis.*
  Nature 590, 89-96 (2021). [doi:10.1038/s41586-021-03213-y](https://doi.org/10.1038/s41586-021-03213-y).
  Files obtained from [b-shields/edbo](https://github.com/b-shields/edbo) (MIT).

Methods compared against / built on:

- *Reasoning BO: Enhancing Bayesian Optimization with Long-Context Reasoning Power of LLMs.*
  [arXiv:2505.12833](https://arxiv.org/abs/2505.12833) (2025) — the IMP@k metric and the
  direct-arylation protocol reproduced here.
- Ramos, M. C. et al. *Bayesian Optimization of Catalysis with In-Context Learning.*
  ACS Cent. Sci. (2026). [doi:10.1021/acscentsci.5c02418](https://doi.org/10.1021/acscentsci.5c02418)
  ([arXiv:2304.05341](https://arxiv.org/abs/2304.05341)).
- González, J., Dai, Z., Hennig, P. & Lawrence, N. *Batch Bayesian Optimization via Local
  Penalization.* AISTATS (2016). [arXiv:1505.08052](https://arxiv.org/abs/1505.08052) —
  the batching used by the classic-BO baseline.

## License

MIT — see [LICENSE](LICENSE). The license covers the code in this repository only; the
reaction datasets belong to their original authors (cited above) and are fetched from
their public sources rather than redistributed.
