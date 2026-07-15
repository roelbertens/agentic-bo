# Deciding the next experiment — RL, BO, and LLM agents for scientific discovery

Three small, self-contained, fully-reproducible studies on one theme: **how to decide the
next experiment or design**, with the evaluation hygiene needed to trust the answer.
Everything runs offline (an LLM agent is optional); every headline number has an exact
reference or a multi-seed spread behind it. Each study lives in its own package under
`src/`; they share only the data loaders (`src/datasets/`).

## 1. Agentic Bayesian optimisation — does putting an LLM in the BO loop help?

Pool-based Bayesian optimisation, where an LLM agent picks the next experiment from a
surrogate-informed shortlist, benchmarked against classic BO across a synthetic materials
task and two **real reaction-optimisation datasets**. On the hard one it reproduces a
published LLM-BO result. The honest finding is that it is **dataset-dependent** — the
agent helps on a deceptive landscape and *hurts* on an easy one — and that an apparent win
is only as trustworthy as the baseline, the metric, and the leakage checks behind it.

→ **[docs/AGENTIC_BO.md](docs/AGENTIC_BO.md)** — how to run, the headline result, the
credibility checks, the genuinely-agentic tool loop, and what the study taught (metric
choice, fair baselines, reproducing Reasoning-BO, leakage probes).
→ **[docs/AGENTIC_BO_METHODS.md](docs/AGENTIC_BO_METHODS.md)** — every policy walked through one identical
decision, side by side.

## 2. Sequential design — reinforcement learning vs. Bayesian optimisation

The other regime: *constructing* a candidate one block at a time under uncertainty, which
is a Markov decision process, not a bandit. It contrasts open- vs. closed-loop control and
model-based vs. model-free learning, and it doubles as an honest **sim-to-real check** —
realism (measurement noise and a 3-body reward) switches on with `--realism`. The lesson: an
optimiser over fixed designs (BO, or an open-loop policy) is capped at the best fixed plan;
only reacting to the realised state reaches the optimum; and under realism **noise only slows
a well-specified model down, while a misspecified model is capped** no matter how much data it
sees — the narrow niche where model-free RL earns its sample cost.

→ **[docs/RL_DESIGN.md](docs/RL_DESIGN.md)** — the task, the methods, both experiments explained panel by
panel, and a decision guide for *when to reach for RL vs. BO vs. a model-based hybrid*.

## 3. Multi-fidelity — when does a cheap simulator pay off?

The same pool task, but with **two oracles**: a cheap simulation whose reliability is a
knob (the LF/HF correlation ρ) and an expensive ground-truth measurement, under one budget
in cost units. The baselines ignore the simulator (classic BO), trust it blindly
(screen-then-confirm), or model it (multi-fidelity BO); against them runs a
[PydanticAI](https://ai.pydantic.dev) agent that spends the budget through typed tools and
must infer from its own simulate-vs-measure pairs whether the simulator can be trusted.
Only measured values count as results; the fidelity mix over time is plotted, not just the
outcome. The finding is that **stating the inference is not the same as acting on it**: the
agent reads the simulator's reliability correctly in its rationales, yet spends a nearly
fixed fidelity mix regardless, and loses to a five-line heuristic applying the same
evidence mechanically. The study doubles as this repo's most complete agentic setup: a typed tool loop
with validation as retries, structured round decisions, fallback handling, a replay cache,
decision-log audits, and a deterministic stand-in model that lets CI exercise the real loop.

→ **[docs/MULTI_FIDELITY.md](docs/MULTI_FIDELITY.md)** — the oracle construction, the
methods, the ρ × cost-ratio sweep, and the agent design (tools, validation-as-retry,
replay cache).

## Quick start

Uses [uv](https://docs.astral.sh/uv/) (no uv? `pip install -e ".[gemini]"`, drop `uv run`).

```bash
uv sync                                   # offline; --extra gemini / --extra mf for the LLM legs

uv run run_agentic_bo.py --dataset buchwald   # study 1: agentic BO on a real reaction dataset
uv run run_rl_design.py                       # study 2: sequential design, RL vs. BO
uv run run_rl_design.py --realism             # the same task with noise + 3-body reality
uv run run_multi_fidelity.py --sweep          # study 3: when does a cheap simulator pay off?
uv run pytest -q                          # the test suite (offline)
```

Full command references live in the docs above. Outputs (plots + summary JSON) land in
`results/<study>/`; the reaction datasets are fetched by `uv run scripts/get_data.py`.

## Layout

One folder per study, sharing only the data loaders:

```
run_agentic_bo.py          CLI study 1: agentic-BO benchmark — plot + summary + decision log
run_rl_design.py           CLI study 2: sequential-design RL vs. BO; --realism for sim-to-real
run_multi_fidelity.py      CLI study 3: two oracles, one budget; --sweep for rho x cost ratio
scripts/get_data.py        download the reaction datasets (Buchwald + arylation)
scripts/audit_decisions.py rationale audit (study 1): the agent's reasoning vs ground truth
scripts/audit_mf_decisions.py rationale audit (study 3): fidelity mix, screen-confirm, loop health
scripts/leakage_probe.py   zero-shot data-leakage probe (with/without reagent names)
scripts/push_to_langfuse.py backfill a decision log into Langfuse (traces + audit scores)
src/datasets/              SHARED: pool container + permute-yields check (base.py),
                           real reaction loaders (reactions.py), synthetic MOF pool (synthetic.py)
src/agentic_bo/            study 1: LLM agents vs classic BO on pool-based optimisation
  surrogate.py             GP + Expected Improvement
  agent.py                 decision context + Heuristic / Gemini / Claude backends (single call)
  agentic.py               genuinely-agentic tool loop: env + tools, two-phase agent, memory
  policies.py              random, classic BO, agentic BO (+ ablation), agentic-tools
  experiment.py            multi-seed runs, convergence curves + per-round IMP@k
  plotting.py, cache.py, tracing.py   figures, prompt->decision cache, optional Langfuse
src/rl_design/             study 2: RL vs. BO on a sequential design MDP
  design.py                linker-design MDP: reward (+ optional 3-body/noise), exact context-c DP
  reinforce.py             from-scratch REINFORCE: one Policy over context 0 (open) / 1 / 2
  methods.py               the roster: policy training, model-based planner, BO over plans
  surrogate.py             GP + EI + Bayesian-linear surrogate (this study's own copy)
src/multi_fidelity/        study 3: cheap simulator vs expensive measurement under one budget
  oracles.py               the corrupted LF oracle (rho knob), cost ledger, curves per unit cost
  campaign.py              campaign state + multi-seed runner
  baselines.py             random / classic BO / two-stage screening / multi-fidelity BO
  agent.py                 PydanticAI agent: typed tools, validation as ModelRetry, rounds
  heuristic.py             deterministic FunctionModel — the offline agent, same loop as the LLM
  cache.py, surrogate.py   LLM replay cache; GP + EI + augmented-input MF fit (own copy)
tests/                     per-study folders, offline end-to-end + credibility checks
docs/AGENTIC_BO.md         study 1: how to run, headline result, credibility checks, tool loop
docs/AGENTIC_BO_METHODS.md study 1: every policy walked through one identical decision
docs/RL_DESIGN.md          study 2: sequential design — task, methods, results, sim-to-real
docs/MULTI_FIDELITY.md     study 3: oracles, methods, sweep, agent design
docs/TESTING.md            the test layers: exact-answer units -> agent loop -> golden -> controls
.claude/skills/            repo skills: quality-gate (pre-commit), experiment-hygiene, writing-style
.github/workflows/ci.yml   CI: ruff + pytest on 3.10 and 3.13
```

## Development

`uv run pytest -q` (offline) and `uv run --group dev ruff check .` must both be clean; CI
enforces them. [docs/TESTING.md](docs/TESTING.md) describes the test layers — exact-answer
unit tests, the deterministically-driven agent loop, golden tests (prompts, parsers, one
pinned trajectory), behavioural controls, and benchmark-level controls. The repo ships three [Claude Code](https://claude.com/claude-code) skills in
`.claude/skills/` — a pre-commit **quality-gate** (tests, lint, no secrets, docs in sync,
prompt/cache stability), **experiment-hygiene** (seeds, metrics, fallback checks), and
**writing-style** (plain explanatory prose in docs and README) — so agent-assisted changes
are held to the same standard as manual ones.

## License

MIT — see [LICENSE](LICENSE). The license covers the code in this repository only; the
reaction datasets belong to their original authors (cited in
[docs/AGENTIC_BO.md](docs/AGENTIC_BO.md#references)) and are fetched from their public
sources rather than redistributed.
