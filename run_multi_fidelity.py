#!/usr/bin/env python3
"""Run the multi-fidelity experiment: when does a cheap simulator pay off?

Examples
--------
# Fast, offline, no API key (deterministic heuristic drives the agent loop):
    uv run run_multi_fidelity.py

# The full sweep over simulator quality x cost ratio (offline):
    uv run run_multi_fidelity.py --sweep

# Real reaction dataset with Gemini as the agent (cached; free to rerun):
    export GEMINI_API_KEY=...
    uv run run_multi_fidelity.py --dataset arylation --agent gemini --seeds 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
# The cost grid is NaN before the first measurement; nanmean over that slice is intended.
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="Degrees of freedom")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from datasets import reactions, synthetic
from multi_fidelity.baselines import MFBO, ClassicBOHF, RandomHF, TwoStage
from multi_fidelity.campaign import MethodResult, run_method
from multi_fidelity.oracles import MultiFidelityProblem

BASELINES = {p.name: p for p in (RandomHF, ClassicBOHF, TwoStage, MFBO)}
ALL_METHODS = [*BASELINES, "agentic_mf"]


def load_dataset(args):
    if args.dataset == "mof":
        return synthetic.sample_pool(n=args.pool_size, seed=args.pool_seed)
    if args.dataset == "buchwald":
        return reactions.load_buchwald_hartwig(subsample=args.subsample, seed=args.pool_seed)
    if args.dataset == "arylation":
        return reactions.load_direct_arylation()
    raise ValueError(args.dataset)


def build_agent_model(args, tag: str):
    """The model that drives AgenticMF: offline heuristic, or Gemini behind the
    replay cache (imports live here so the baselines run without the mf extra)."""
    if args.agent == "heuristic":
        from multi_fidelity.heuristic import heuristic_model

        return heuristic_model()
    from pydantic_ai.models.google import GoogleModel

    model = GoogleModel(args.model)
    if args.no_cache:
        return model
    from multi_fidelity.cache import CachedModel

    return CachedModel(model, os.path.join(args.out, "mf_llm_cache", f"{tag}.json"))


def run_config(dataset, args, rho: float, cost_hf: float, tag: str,
               verbose: bool) -> dict[str, MethodResult]:
    problem = MultiFidelityProblem(dataset, rho=rho, cost_lf=args.cost_lf,
                                   cost_hf=cost_hf, seed=args.pool_seed)
    seeds = range(args.seeds)
    results: dict[str, MethodResult] = {}
    agent_logs = []
    for name in args.methods:
        if name in BASELINES:
            make = BASELINES[name]
        else:
            from multi_fidelity.agent import AgenticMF

            model = build_agent_model(args, tag)

            def make(model=model):
                policy = AgenticMF(model, request_limit=args.request_limit,
                                   verbose=args.verbose)
                agent_logs.append(policy)
                return policy
        results[name] = run_method(problem, make, name, seeds, args.cost_budget,
                                   args.n_init, verbose=verbose)
    if agent_logs:
        path = os.path.join(args.out, f"decisions_{tag}.jsonl")
        with open(path, "w") as fh:
            for seed, policy in enumerate(agent_logs):
                for rec in policy.decision_log:
                    fh.write(json.dumps({"seed": seed, **rec}) + "\n")
        print(f"  agent decision log -> {path}")
    return results


def plot(results: dict[str, MethodResult], problem_label: str, budget: float, path: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = np.arange(int(budget) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, res in results.items():
        (line,) = ax1.plot(grid, res.mean, label=name)
        ax1.fill_between(grid, res.mean - res.std, res.mean + res.std,
                         alpha=0.15, color=line.get_color())
        ax2.plot(grid, np.nanmean(res.hf_share, axis=0), color=line.get_color(), label=name)
    ax1.axhline(next(iter(results.values())).best_possible, ls=":", c="gray", lw=1)
    ax1.set(xlabel="cumulative cost", ylabel="best measured value",
            title=f"Convergence per unit cost — {problem_label}")
    ax2.set(xlabel="cumulative cost", ylabel="fraction of spend on measurements",
            ylim=(0, 1.05), title="Fidelity mix over time")
    ax1.legend()
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"  plot -> {path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=["mof", "buchwald", "arylation"], default="mof")
    p.add_argument("--rho", type=float, default=0.8,
                   help="simulator quality: LF/HF correlation in [0, 1] (default 0.8)")
    p.add_argument("--cost-lf", type=float, default=1.0, help="cost of one simulation")
    p.add_argument("--cost-hf", type=float, default=10.0, help="cost of one measurement")
    p.add_argument("--cost-budget", type=float, default=200.0,
                   help="total campaign budget in cost units (default 200)")
    p.add_argument("--n-init", type=int, default=3,
                   help="random initial measurements, charged to the budget (default 3)")
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--methods", nargs="+", default=ALL_METHODS, choices=ALL_METHODS)
    p.add_argument("--agent", choices=["heuristic", "gemini"], default="heuristic",
                   help="what drives agentic_mf (default: heuristic, offline)")
    p.add_argument("--model", default="gemini-2.5-flash")
    p.add_argument("--request-limit", type=int, default=24,
                   help="max LLM requests per agent round (default 24)")
    p.add_argument("--no-cache", action="store_true", help="disable the LLM replay cache")
    p.add_argument("--sweep", action="store_true",
                   help="grid over rho x cost ratio instead of a single run; prints the "
                        "when-does-multi-fidelity-pay-off table")
    p.add_argument("--pool-size", type=int, default=600)
    p.add_argument("--pool-seed", type=int, default=0)
    p.add_argument("--subsample", type=int, default=None)
    p.add_argument("--out", default="results/multi_fidelity")
    p.add_argument("--verbose", action="store_true", help="print each agent round's rationale")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    dataset = load_dataset(args)
    summary: dict = {"config": vars(args) | {"dataset_n": dataset.n}}

    if args.sweep:
        rhos, ratios = (0.9, 0.6, 0.3), (5.0, 10.0, 20.0)
        table: dict[str, dict[str, float]] = {m: {} for m in args.methods}
        for rho in rhos:
            for ratio in ratios:
                tag = f"mf_{args.dataset}_{args.agent}_rho{rho:g}_r{ratio:g}"
                print(f"rho={rho:g} cost ratio={ratio:g}")
                results = run_config(dataset, args, rho, args.cost_lf * ratio, tag,
                                     verbose=False)
                for m, res in results.items():
                    table[m][f"rho={rho:g},ratio={ratio:g}"] = float(res.final_pct.mean())
        width = max(map(len, args.methods)) + 2
        cols = list(next(iter(table.values())))
        print("\nFinal best measurement, % of pool optimum "
              f"(mean over {args.seeds} seeds)\n")
        print(" " * width + "  ".join(f"{c:>16s}" for c in cols))
        for m in args.methods:
            print(f"{m:<{width}s}" + "  ".join(f"{table[m][c]:16.1f}" for c in cols))
        summary["sweep"] = table
        out = os.path.join(args.out, f"mf_{args.dataset}_{args.agent}_sweep.json")
    else:
        tag = f"mf_{args.dataset}_{args.agent}_rho{args.rho:g}_r{args.cost_hf / args.cost_lf:g}"
        results = run_config(dataset, args, args.rho, args.cost_hf, tag, verbose=True)
        label = (f"{args.dataset}, rho={args.rho:g}, "
                 f"cost {args.cost_hf:g}:{args.cost_lf:g}, budget {args.cost_budget:g}")
        plot(results, label, args.cost_budget, os.path.join(args.out, f"{tag}.png"))
        summary["final_pct"] = {m: {"mean": float(r.final_pct.mean()),
                                    "std": float(r.final_pct.std())}
                                for m, r in results.items()}
        summary["hf_share_final"] = {m: float(np.nanmean(r.hf_share[:, -1]))
                                     for m, r in results.items()}
        for m, s in summary["final_pct"].items():
            print(f"  {m:<16s} final {s['mean']:5.1f} +/- {s['std']:4.1f} % of optimum")
        out = os.path.join(args.out, f"{tag}_summary.json")

    with open(out, "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(f"  summary -> {out}")


if __name__ == "__main__":
    main()
