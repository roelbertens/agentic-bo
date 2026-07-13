# Deciding the next experiment — RL, BO, and LLM agents for scientific discovery

Two small, self-contained, fully-reproducible studies on one theme: **how to decide the
next experiment or design**, with the evaluation hygiene needed to trust the answer.
Everything runs offline (an LLM agent is optional); every headline number has an exact
reference or a multi-seed spread behind it.

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
→ **[docs/METHODS.md](docs/METHODS.md)** — every policy walked through one identical
decision, side by side.

## 2. Sequential design — reinforcement learning vs. Bayesian optimisation

The other regime: *constructing* a candidate one block at a time under uncertainty, which
is a Markov decision process, not a bandit. It contrasts open- vs. closed-loop control and
model-based vs. model-free learning, and it doubles as an honest **sim-to-real check** —
realism (measurement noise and a 3-body reward) is a dial (`--realism`). The lesson: an
optimiser over fixed designs (BO, or an open-loop policy) is capped at the best fixed plan;
only reacting to the realised state reaches the optimum; and under realism **noise only slows
a well-specified model down, while a misspecified model is capped** no matter how much data it
sees — the narrow niche where model-free RL earns its sample cost.

→ **[docs/RL.md](docs/RL.md)** — the task, the methods, both experiments explained panel by
panel, and a decision guide for *when to reach for RL vs. BO vs. a model-based hybrid*.

## Quick start

Uses [uv](https://docs.astral.sh/uv/) (no uv? `pip install -e ".[gemini]"`, drop `uv run`).

```bash
uv sync                                   # offline; add --extra gemini for the LLM agent

uv run run.py --dataset buchwald          # agentic BO on a real reaction dataset (offline)
uv run rl_design.py                       # sequential design: RL vs. BO (offline)
uv run rl_design.py --realism             # the same task with noise + 3-body reality
uv run pytest -q                          # the test suite (offline)
```

Full command references live in the two docs above. Outputs (plots + summary JSON) land in
`results/`; the reaction datasets are fetched by `uv run scripts/get_data.py`.

## Layout

```
run.py                     CLI: agentic-BO benchmark — plot + summary + decision log
rl_design.py               CLI: sequential-design RL vs. BO; --realism for the sim-to-real gap
scripts/get_data.py        download the reaction datasets (Buchwald + arylation)
scripts/audit_decisions.py rationale audit: check the agent's reasoning against ground truth
scripts/leakage_probe.py   zero-shot data-leakage probe (with/without reagent names)
scripts/push_to_langfuse.py backfill a decision log into Langfuse (traces + audit scores)
src/agentic_bo/
  data.py                  dataset-agnostic container + permute-yields leakage check
  objective.py             synthetic MOF pool + ground-truth objective
  reactions.py             real loaders: Buchwald-Hartwig + direct arylation (named reagents)
  surrogate.py             GP + Expected Improvement, and a Bayesian-linear surrogate
  agent.py                 decision context + Heuristic / Gemini / Claude backends (single call)
  agentic.py               genuinely-agentic tool loop: env + tools, two-phase agent, memory
  policies.py              random, classic BO, agentic BO (+ ablation), agentic-tools
  experiment.py            multi-seed runs, convergence curves + per-round IMP@k
  plotting.py              convergence + regret figures
  cache.py                 persistent prompt->decision cache (free resume)
  tracing.py               optional Langfuse tracing (no-op unless --langfuse)
  design.py                linker-design MDP: reward (+ optional 3-body/noise), exact context-c DP
  reinforce.py             from-scratch REINFORCE: one Policy over context 0 (open) / 1 / 2
  methods.py               RL-vs-BO roster: policy training, model-based planner, BO over plans
tests/                     end-to-end + credibility-tooling checks (offline)
docs/AGENTIC_BO.md         study 1: how to run, headline result, credibility checks, tool loop
docs/RL.md                 study 2: sequential design — task, methods, results, sim-to-real
docs/METHODS.md            every BO policy on one identical decision, side by side
.claude/skills/            repo skills: quality-gate (pre-commit) + experiment-hygiene
.github/workflows/ci.yml   CI: ruff + pytest on 3.10 and 3.13
```

## Development

`uv run pytest -q` (offline) and `uv run --group dev ruff check .` must both be clean; CI
enforces them. The repo ships two [Claude Code](https://claude.com/claude-code) skills in
`.claude/skills/` — a pre-commit **quality-gate** (tests, lint, no secrets, docs in sync,
prompt/cache stability) and **experiment-hygiene** (seeds, metrics, fallback checks) — so
agent-assisted changes are held to the same standard as manual ones.

## License

MIT — see [LICENSE](LICENSE). The license covers the code in this repository only; the
reaction datasets belong to their original authors (cited in
[docs/AGENTIC_BO.md](docs/AGENTIC_BO.md#references)) and are fetched from their public
sources rather than redistributed.
