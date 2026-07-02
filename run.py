#!/usr/bin/env python3
"""Run the agentic Bayesian-optimisation benchmark and produce plots + a summary.

Examples
--------
# Fast, offline, no API key (heuristic agent stands in for the LLM):
    python run.py

# Real reaction dataset where an LLM's prior knowledge can help, with Gemini:
    export GEMINI_API_KEY=...
    python run.py --dataset buchwald --agent gemini --seeds 3 --budget 20 --verbose

# Same task with Claude instead:
    export ANTHROPIC_API_KEY=...
    python run.py --dataset buchwald --agent claude --seeds 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from agentic_bo import objective, plotting, reactions
from agentic_bo.experiment import run_method, run_method_per_substrate
from agentic_bo.policies import AgenticBO, ClassicBO, RandomPolicy

ALL_METHODS = ["random", "classic_bo", "agentic_bo", "agentic_no_surrogate"]


def build_agent(kind: str, model: str, verbose: bool, cache=None):
    if kind == "heuristic":
        from agentic_bo.agent import HeuristicAgent

        return HeuristicAgent()
    if kind == "gemini":
        from agentic_bo.agent import GeminiAgent

        return GeminiAgent(model=model, verbose=verbose, cache=cache)
    if kind == "claude":
        from agentic_bo.agent import AnthropicAgent

        return AnthropicAgent(model=model, verbose=verbose, cache=cache)
    raise ValueError(f"unknown agent kind: {kind}")


DEFAULT_MODEL = {"gemini": "gemini-2.5-flash", "claude": "claude-opus-4-8", "heuristic": ""}


def load_dataset(args):
    if args.dataset == "mof":
        return objective.sample_pool(n=args.pool_size, seed=args.pool_seed)
    if args.dataset == "buchwald":
        return reactions.load_buchwald_hartwig(subsample=args.subsample, seed=args.pool_seed)
    if args.dataset == "arylation":
        return reactions.load_direct_arylation()
    raise ValueError(args.dataset)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=["mof", "buchwald", "arylation"], default="mof",
                   help="mof = smooth synthetic; buchwald = easy real reaction pool; "
                        "arylation = harder, deceptive real reaction pool")
    p.add_argument("--budget", type=int, default=20, help="experiments after init (default 20)")
    p.add_argument("--n-init", type=int, default=5, help="random initial evaluations (default 5)")
    p.add_argument("--batch-size", type=int, default=1,
                   help="candidates proposed per round (q). Set 3 to match Reasoning-BO's protocol")
    p.add_argument("--seeds", type=int, default=8, help="number of random seeds (default 8)")
    p.add_argument("--pool-size", type=int, default=600, help="candidate pool size (mof only)")
    p.add_argument("--pool-seed", type=int, default=0, help="seed for the (fixed) candidate pool")
    p.add_argument("--subsample", type=int, default=None, help="subsample the buchwald pool (optional)")
    p.add_argument("--per-substrate", action="store_true",
                   help="buchwald: fix the aryl halide and optimise 4x3x22, averaged over substrates")
    p.add_argument("--substrates", type=int, default=4,
                   help="how many substrates to average over in --per-substrate mode")
    p.add_argument("--agent", choices=["heuristic", "gemini", "claude"], default="heuristic",
                   help="backend for the agentic policy (default: heuristic, needs no API key)")
    p.add_argument("--model", default=None, help="LLM model id (defaults per agent)")
    p.add_argument("--methods", nargs="+", default=ALL_METHODS, choices=ALL_METHODS)
    p.add_argument("--out", default="results", help="output directory")
    p.add_argument("--no-cache", action="store_true",
                   help="disable the persistent decision cache (LLM agents resume for free by default)")
    p.add_argument("--verbose", action="store_true", help="print each agent decision's rationale")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    seeds = list(range(args.seeds))
    model = args.model or DEFAULT_MODEL[args.agent]
    per_sub = args.per_substrate
    if per_sub and args.dataset != "buchwald":
        p.error("--per-substrate only applies to --dataset buchwald")

    n_agentic = sum(m.startswith("agentic") for m in args.methods)
    import math
    rounds = math.ceil(args.budget / args.batch_size)
    if per_sub:
        substrate_sets = reactions.load_buchwald_by_substrate(limit=args.substrates)
        plot_title = f"Buchwald-Hartwig per-substrate (avg over {len(substrate_sets)})"
        plot_objective = "% of substrate optimum"
        pool_desc = f"{len(substrate_sets)} substrates x {substrate_sets[0].n} candidates"
        optimum_display = 100.0
        n_decisions = len(substrate_sets) * args.seeds * rounds * n_agentic
    else:
        dataset = load_dataset(args)
        plot_title = dataset.title
        plot_objective = dataset.objective_label
        pool_desc = f"{dataset.n} candidates | global optimum = {dataset.best_value:.2f}"
        optimum_display = dataset.best_value
        n_decisions = args.seeds * rounds * n_agentic

    print(f"Dataset: {plot_title}")
    print(f"Pool: {pool_desc}")
    print(f"Budget: {args.n_init} init + {args.budget} experiments "
          f"(batch {args.batch_size} -> {rounds} rounds) | seeds: {args.seeds} | agent: {args.agent}")
    if args.agent != "heuristic" and n_decisions:
        print(f"Heads-up: up to ~{n_decisions} {args.agent} API calls this run.")
    print()

    agent = None
    if n_agentic:
        cache = None
        if args.agent != "heuristic" and not args.no_cache:
            from agentic_bo.cache import PromptCache

            cache_path = os.path.join(".cache", f"decisions_{args.agent}_{model}.json")
            cache = PromptCache(cache_path)
            if cache.data:
                print(f"Decision cache: {len(cache.data)} entries at {cache_path} "
                      f"(matching steps resume for free).")
        agent = build_agent(args.agent, model, args.verbose, cache=cache)

    factories = {
        "random": lambda: RandomPolicy(),
        "classic_bo": lambda: ClassicBO(),
        "agentic_bo": lambda: AgenticBO(agent, use_surrogate=True),
        "agentic_no_surrogate": lambda: AgenticBO(agent, use_surrogate=False),
    }

    results = []
    for name in args.methods:
        print(f"Running {name} ...")
        if per_sub:
            results.append(run_method_per_substrate(
                substrate_sets, factories[name], name, seeds, args.budget, args.n_init,
                args.batch_size))
        else:
            results.append(run_method(
                dataset, factories[name], name, seeds, args.budget, args.n_init,
                args.batch_size))

    # Evaluation hygiene: if the agent fell back to EI/heuristic, the "agentic" numbers
    # are not really the model — make that impossible to miss.
    if agent is not None and getattr(agent, "calls", 0):
        hits = getattr(agent, "cache_hits", 0)
        api_calls = agent.calls - hits
        cache_note = f" ({hits} from cache, {api_calls} live API calls)" if hits else ""
        fb = agent.fallbacks
        if fb:
            print(f"\n[!] {args.agent} agent fell back {fb}/{agent.calls} decisions "
                  f"({100 * fb / agent.calls:.0f}%). Those picks are EI/heuristic, not the model.{cache_note}")
        else:
            print(f"\n[ok] {args.agent} agent made all {agent.calls} decisions "
                  f"(no fallbacks).{cache_note}")

    agent_label = args.agent if args.agent == "heuristic" else f"{args.agent}:{model}"
    tag = f"{args.dataset}{'_persub' if per_sub else ''}_{args.agent}"
    plot_path = os.path.join(args.out, f"convergence_{tag}.png")
    plotting.plot_convergence(results, agent_label, args.n_init, plot_path,
                              objective_label=plot_objective, title=plot_title)

    nrounds = results[0].round_quality.shape[1]
    ncol = results[0].curves.shape[1]

    # Early checkpoints. In batch mode we match Reasoning-BO's IMP@k = the per-round
    # PROPOSAL quality (best yield among round k's batch) — non-cumulative, so it can go
    # down, exactly like the paper's table. In single-step mode we fall back to
    # best-so-far after k steps. Values are in the objective's own units (raw yield %).
    if args.batch_size > 1:
        cps = [(f"IMP@{k}", ("round", k - 1)) for k in (1, 3, 5) if k <= nrounds]
    else:
        cps = [(f"@{s}", ("curve", min(args.n_init + s - 1, ncol - 1)))
               for s in (5, 10) if s <= args.budget]

    def value_at(res, ref):
        kind, i = ref
        arr = res.round_quality if kind == "round" else res.curves
        return float(arr[:, i].mean())

    metric_note = ("IMP@k = per-round proposal quality (matches Reasoning-BO)"
                   if args.batch_size > 1 else "@k = best-so-far after k steps")
    print(f"\n=== Summary (final = best-so-far; {metric_note}) ===")
    head = f"{'method':<26} {'final':>9} {'%opt':>6}"
    for lbl, _ in cps:
        head += f" {lbl:>8}"
    print(head)
    summary = {"dataset": plot_title, "optimum": optimum_display, "agent": agent_label,
               "batch_size": args.batch_size, "checkpoints": [c[0] for c in cps], "methods": {}}
    for res in results:
        final = res.curves[:, -1]
        pct = 100 * final.mean() / res.best_possible
        row = f"{res.name:<26} {final.mean():>6.1f}±{final.std():>2.0f} {pct:>5.0f}%"
        entry = {"final_best_mean": float(final.mean()), "final_best_std": float(final.std()),
                 "pct_of_optimum_final": float(pct)}
        for lbl, ref in cps:
            v = value_at(res, ref)
            row += f" {v:>8.1f}"
            entry[lbl] = v
        print(row)
        summary["methods"][res.name] = entry

    with open(os.path.join(args.out, f"summary_{tag}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved plot -> {plot_path}")
    print(f"Saved summary -> {os.path.join(args.out, f'summary_{tag}.json')}")


if __name__ == "__main__":
    main()
