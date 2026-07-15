#!/usr/bin/env python3
"""Design molecular linkers on one task — RL vs. BO — with realism as a dial.

One setup (``rl_design/design.py``): a stochastic sequential MDP where a linker is
built block by block and a verifier scores the realised chain. The same methods
(``rl_design/methods.py``) run at two settings of the reality knobs:

* **idealised** (default) — pairwise reward, no noise. Shows the open-vs-closed-loop
  ceiling and where BO fits: fixed-plan optimisers (BO, open-loop RL) are capped at the
  best fixed plan; closing the loop (RL, or a model-based Bayesian planner) reaches the
  optimum, the planner far more sample-efficiently.
* **realistic** (``--realism``) — adds strong measurement noise and a 3-body reward a
  pairwise model cannot represent. Shows the sim-to-real gap: a misspecified model caps a
  planner short of the optimum no matter how many episodes it sees (and no amount of extra
  state repairs it), a well-specified model still reaches the optimum an order of magnitude
  sooner than model-free RL, and model-free RL is the slow-but-safe route that needs no
  model at all — only enough state.

Exact reference values (backward DP) set an honest "% of optimum" scale in both.

    uv run rl_design.py                 # idealised
    uv run rl_design.py --realism       # noisy + 3-body reality
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from rl_design.design import DesignTask
from rl_design.methods import bo_on_plans, model_based_planner, train_policy_curve

_IDEAL_COLORS = {"BO over plans (naive)": "#c5b0d5", "BO over plans (structured)": "#1f77b4",
                 "open-loop RL": "#ff7f0e", "closed-loop RL": "#2ca02c",
                 "model-based Bayesian planner": "#d62728"}
_REAL_COLORS = {"Bayesian planner (naive: pairwise, 1-block)": "#d62728",
                "closed-loop RL (1-block state)": "#2ca02c",
                "Bayesian planner (reality-aware: 3-body, 2-block)": "#9467bd",
                "closed-loop RL (2-block state)": "#1f77b4"}


def _smooth(values):
    """Trailing moving average (adaptive window) — stabilises the jittery model-based
    planner curves under heavy noise so the reported plateau and the plot are steady."""
    values = np.asarray(values, dtype=float)
    k = max(3, len(values) // 25)
    csum = np.cumsum(np.insert(values, 0, 0.0))
    idx = np.arange(1, len(values) + 1)
    lo = np.maximum(0, idx - k)
    return (csum[idx] - csum[lo]) / (idx - lo)


def _panel(ax, curves, colors, title, lines):
    for name, (s, v) in curves.items():
        ax.plot(s, v, color=colors[name], lw=2, marker="o", ms=3, label=name)
    for value, style, colour, alpha, label in lines:
        ax.axhline(value, ls=style, color=colour, lw=1.2, alpha=alpha, label=label)
    ax.set_xscale("log")
    ax.set_xlabel("episodes seen  (verifier calls, log scale)")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)


def _plot(left, right, left_title, right_title, colors, lines_l, lines_r, ylabel, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(13.5, 5.2), sharey=True)
    _panel(axl, left, colors, left_title, lines_l)
    _panel(axr, right, colors, right_title, lines_r)
    axl.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def run_idealised(task, args, rng_of, plot_path):
    """Five methods on the clean task: the open/closed-loop ceiling and BO's role.

    The RL curves run first: their (early-stopped) episode counts set the other methods'
    budgets, so every curve in a panel spans the same episode range — a method that
    plateaus early is *shown* staying flat, not cut off. (BO is capped at 400 evaluations
    in case open-loop RL ever runs its full budget: refitting the naive GP thousands of
    times is not worth plotting more of an already-flat line.)"""
    closed_opt, open_opt = task.optimal_value(1), task.open_loop_optimum()
    olr = train_policy_curve(task, 0, rng_of(), args.batch, args.eval_every,
                             args.max_updates, 0.5, 0.02, target(task, open_opt))
    clr = train_policy_curve(task, 1, rng_of(), args.batch, args.eval_every,
                             args.max_updates, 0.5, 0.02, target(task, closed_opt))
    bo_iters = min(400, int(np.ceil(olr[0][-1] / args.bo_replicates))) - 6
    bon = bo_on_plans(task, rng_of(), 6, bo_iters, args.bo_replicates, False)
    bos = bo_on_plans(task, rng_of(), 6, bo_iters, args.bo_replicates, True)
    mbp = model_based_planner(task, rng_of(), int(clr[0][-1]), order=2, context=1)
    raw = [("BO over plans (naive)", *bon), ("BO over plans (structured)", *bos),
           ("open-loop RL", *olr), ("closed-loop RL", *clr),
           ("model-based Bayesian planner", *mbp)]
    rows = [(k, s, _smooth(v) if "RL" not in k else v) for k, s, v in raw]
    left = {k: (s, v) for k, s, v in rows[:3]}
    right = {k: (s, v) for k, s, v in rows[3:]}
    lines_l = [(open_opt, "-.", "#8c564b", 0.9, f"best fixed plan ({open_opt:.2f})"),
               (closed_opt, "--", "k", 0.5, f"optimal policy ({closed_opt:.2f})"),
               (task.random_value(), ":", "grey", 0.7, "")]
    lines_r = [(open_opt, "-.", "#8c564b", 0.5, f"best fixed plan ({open_opt:.2f})"),
               (closed_opt, "--", "k", 0.9, f"optimal policy ({closed_opt:.2f})"),
               (task.random_value(), ":", "grey", 0.7, "")]
    _plot(left, right, "Open loop — reaching the fixed-plan ceiling",
          "Closed loop — breaking the ceiling", _IDEAL_COLORS, lines_l, lines_r,
          "expected reward of the deployed policy/plan", plot_path)
    return rows, {"optimal_policy": closed_opt, "best_fixed_plan": open_opt}


def run_realistic(task, args, rng_of, plot_path):
    """Four methods on the noisy 3-body task: what a wrong model costs, what a right one buys.

    Every method trains against the same unreachable-except-by-convergence target (99% of
    the true optimum), so where a curve flattens is its honest ceiling — not an early-stop
    artifact. No "best 1-block policy" reference line: the 1-block DP value (planned with
    the pairwise part of the reward) is only a *lower bound* on the best 1-block policy,
    and learned methods beat it, so it would mislead as a plotted ceiling."""
    opt2 = task.optimal_value(2)
    rl1 = train_policy_curve(task, 1, rng_of(), args.batch, args.eval_every,
                             args.max_updates, 0.5, 0.02, target(task, opt2))
    rl2 = train_policy_curve(task, 2, rng_of(), args.batch, args.eval_every,
                             args.max_updates, 0.5, 0.02, target(task, opt2))
    # RL first: each panel's planner runs to its RL curve's endpoint, so both lines span
    # the same episode range and a plateau is shown staying flat, not cut off.
    naive = model_based_planner(task, rng_of(), int(rl1[0][-1]), order=2, context=1,
                                warmup=120)
    ra = model_based_planner(task, rng_of(), int(rl2[0][-1]), order=3, context=2,
                             warmup=200)
    raw = [("Bayesian planner (naive: pairwise, 1-block)", *naive),
           ("closed-loop RL (1-block state)", *rl1),
           ("Bayesian planner (reality-aware: 3-body, 2-block)", *ra),
           ("closed-loop RL (2-block state)", *rl2)]
    rows = [(k, s, _smooth(v) if "RL" not in k else v) for k, s, v in raw]
    left = {k: (s, v) for k, s, v in rows[:2]}
    right = {k: (s, v) for k, s, v in rows[2:]}
    lines = [(opt2, "--", "k", 0.85, f"true optimum ({opt2:.2f})"),
             (task.random_value(), ":", "grey", 0.7, "")]
    _plot(left, right, "Wrong model or too little state — capped below the optimum",
          "Right model & state — the planner reaches the optimum far sooner than RL",
          _REAL_COLORS, lines, lines, "expected reward of the deployed policy", plot_path)
    return rows, {"true_optimum": opt2}


def target(task, value):
    rnd = task.random_value()
    return rnd + 0.99 * (value - rnd)


def plateau(values):
    """The value a method converges to — the mean of the last quarter of its curve, so a
    jittery planner under noise is not judged by a single unlucky end point."""
    return float(np.mean(values[-max(3, len(values) // 4):]))


def reaches_at(samples, values, final, optimum, rnd):
    """The episode a method first reaches its result (``final``). A method that reaches the
    optimum is measured against the optimum (so the number matches the plot's full climb, the
    slow last percent included); a method that plateaus below is measured against its own
    plateau. Uses the same 99%-of-span threshold as the early stop, so table and plot agree."""
    def tgt(value):
        return rnd + 0.99 * (value - rnd)
    threshold = tgt(optimum) if final >= tgt(optimum) else tgt(final)
    for s, v in zip(samples, values, strict=True):
        if v >= threshold:
            return int(s)
    return int(samples[-1])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--realism", action="store_true",
                   help="turn on measurement noise + a 3-body reward (the sim-to-real gap)")
    p.add_argument("--triplet-strength", type=float, default=1.0,
                   help="3-body reward strength when --realism is set")
    p.add_argument("--obs-noise", type=float, default=3.0,
                   help="measurement-noise std when --realism is set")
    p.add_argument("--length", type=int, default=6)
    p.add_argument("--success-prob", type=float, default=0.6)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--eval-every", type=int, default=8)
    p.add_argument("--max-updates", type=int, default=4000)
    p.add_argument("--bo-replicates", type=int, default=20)
    p.add_argument("--seed", type=int, default=1,
                   help="seed (0 traps the open-loop policy in a local optimum; 1 is typical)")
    p.add_argument("--out", default="results/rl_design")
    args = p.parse_args()

    task = DesignTask.default(
        length=args.length, success_prob=args.success_prob,
        triplet_strength=args.triplet_strength if args.realism else 0.0,
        obs_noise=args.obs_noise if args.realism else 0.0)
    rnd = task.random_value()
    span = (task.optimal_value(2) if args.realism else task.optimal_value(1)) - rnd

    def pct(v):
        return 100 * (v - rnd) / span

    def rng_of():
        return np.random.default_rng(args.seed)

    mode = "realistic (noise + 3-body)" if args.realism else "idealised"
    tag = "_realism" if args.realism else ""
    plot_name = f"rl_design{tag}_curve.png" if args.realism else "rl_design_learning_curve.png"
    summary_name = f"rl_design{tag}_summary.json"
    plot_path = os.path.join(args.out, plot_name)
    print(f"Linker design [{mode}]: length {task.length}, {task.n_blocks} blocks, "
          f"success prob {task.success_prob}")
    os.makedirs(args.out, exist_ok=True)
    runner = run_realistic if args.realism else run_idealised
    rows, refs = runner(task, args, rng_of, plot_path)

    optimum = refs["true_optimum"] if args.realism else refs["optimal_policy"]
    print(f"\n{'method':<42} {'% of optimum':>13} {'reaches it at':>15}")
    methods = {}
    for name, s, v in rows:
        # Under realism the planner curves are jittery, so their result is a plateau mean;
        # RL curves (monotone, early-stopping) and everything in the noise-free idealised
        # task use their final value.
        final = plateau(v) if (args.realism and "RL" not in name) else float(v[-1])
        eps = reaches_at(s, v, final, optimum, rnd)
        print(f"{name:<42} {pct(final):>12.0f}% {eps:>12,} ep")
        methods[name] = {"final": final, "pct_of_optimum": pct(final), "reaches_at": eps}

    summary = {"mode": mode, "seed": args.seed, "random": rnd, **refs, "methods": methods}
    with open(os.path.join(args.out, summary_name), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved plot -> {plot_path}")
    print(f"Saved summary -> {os.path.join(args.out, summary_name)}")


if __name__ == "__main__":
    main()
