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
from agentic_bo.data import permute_yields
from agentic_bo.experiment import run_method, run_method_per_substrate
from agentic_bo.policies import AgenticBO, AgenticToolBO, ClassicBO, RandomPolicy

# "agentic_tools" (the genuinely-agentic tool loop) is opt-in: not in the default set,
# since it needs a tool-capable agent (heuristic offline, or gemini with an API key).
ALL_METHODS = ["random", "classic_bo", "agentic_bo", "agentic_no_surrogate"]
METHOD_CHOICES = [*ALL_METHODS, "agentic_tools"]


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


def build_tool_agent(kind: str, worker_model: str, reasoner_model: str, verbose: bool, cache=None):
    if kind == "heuristic":
        from agentic_bo.agentic import HeuristicToolAgent

        return HeuristicToolAgent()
    if kind == "gemini":
        from agentic_bo.agentic import GeminiToolAgent

        return GeminiToolAgent(worker_model=worker_model, reasoner_model=reasoner_model,
                               verbose=verbose, cache=cache)
    raise ValueError("agentic_tools supports --agent heuristic or gemini (not claude yet)")


DEFAULT_MODEL = {"gemini": "gemini-2.5-flash", "claude": "claude-opus-4-8", "heuristic": ""}


def load_dataset(args):
    if args.dataset == "mof":
        return objective.sample_pool(n=args.pool_size, seed=args.pool_seed)
    if args.dataset == "buchwald":
        return reactions.load_buchwald_hartwig(subsample=args.subsample, seed=args.pool_seed,
                                               anonymize=args.anonymize)
    if args.dataset == "arylation":
        return reactions.load_direct_arylation(anonymize=args.anonymize)
    raise ValueError(args.dataset)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
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
    p.add_argument("--subsample", type=int, default=None,
                   help="subsample the buchwald pool (optional)")
    p.add_argument("--per-substrate", action="store_true",
                   help="buchwald: fix the aryl halide and optimise 4x3x22, "
                        "averaged over substrates")
    p.add_argument("--substrates", type=int, default=4,
                   help="how many substrates to average over in --per-substrate mode")
    p.add_argument("--anonymize", action="store_true",
                   help="name ablation (credibility check): withhold reagent names/SMILES from "
                        "the agent prompt (reaction datasets only). If the cold-start edge "
                        "disappears, it came from named chemistry knowledge")
    p.add_argument("--permute-yields", action="store_true",
                   help="leakage check: shuffle the yields across candidates (seeded by "
                        "--pool-seed). Chemistry knowledge can no longer help, so the agent "
                        "should drop to random; audit the decision log for memorisation")
    p.add_argument("--agent", choices=["heuristic", "gemini", "claude"], default="heuristic",
                   help="backend for the agentic policy (default: heuristic, needs no API key)")
    p.add_argument("--model", default=None, help="LLM model id (defaults per agent)")
    p.add_argument("--worker-model", default="gemini-2.5-flash",
                   help="agentic_tools: model for the investigation step (tool exploration)")
    p.add_argument("--reasoner-model", default="gemini-2.5-flash",
                   help="agentic_tools: model for the decision step. Set a stronger model here "
                        "(e.g. gemini-2.5-pro) to test whether a better reasoner at the decision "
                        "step, with a cheap worker, beats the single-call agent")
    p.add_argument("--methods", nargs="+", default=ALL_METHODS, choices=METHOD_CHOICES)
    p.add_argument("--out", default="results", help="output directory")
    p.add_argument("--no-cache", action="store_true",
                   help="disable the persistent decision cache "
                        "(LLM agents resume for free by default)")
    p.add_argument("--langfuse", action="store_true",
                   help="push every agentic decision to Langfuse as traces + scores "
                        "(needs `uv sync --extra eval` and LANGFUSE_* env vars)")
    p.add_argument("--verbose", action="store_true", help="print each agent decision's rationale")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    seeds = list(range(args.seeds))
    model = args.model or DEFAULT_MODEL[args.agent]
    per_sub = args.per_substrate
    if per_sub and args.dataset != "buchwald":
        p.error("--per-substrate only applies to --dataset buchwald")
    if args.anonymize and args.dataset == "mof":
        p.error("--anonymize only applies to the reaction datasets (buchwald / arylation)")

    want_tools = "agentic_tools" in args.methods
    n_single_agentic = sum(m in ("agentic_bo", "agentic_no_surrogate") for m in args.methods)
    n_agentic = n_single_agentic + want_tools
    if want_tools and args.agent == "claude":
        p.error("agentic_tools supports --agent heuristic or gemini (not claude yet)")
    import math
    rounds = math.ceil(args.budget / args.batch_size)
    if per_sub:
        substrate_sets = reactions.load_buchwald_by_substrate(limit=args.substrates,
                                                              anonymize=args.anonymize)
        if args.permute_yields:
            substrate_sets = [permute_yields(d, seed=args.pool_seed + i)
                              for i, d in enumerate(substrate_sets)]
        plot_title = f"Buchwald-Hartwig per-substrate (avg over {len(substrate_sets)})"
        plot_objective = "% of substrate optimum"
        pool_desc = f"{len(substrate_sets)} substrates x {substrate_sets[0].n} candidates"
        optimum_display = 100.0
        n_decisions = len(substrate_sets) * args.seeds * rounds * n_agentic
    else:
        dataset = load_dataset(args)
        if args.permute_yields:
            dataset = permute_yields(dataset, seed=args.pool_seed)
        plot_title = dataset.title
        plot_objective = dataset.objective_label
        pool_desc = f"{dataset.n} candidates | global optimum = {dataset.best_value:.2f}"
        optimum_display = dataset.best_value
        n_decisions = args.seeds * rounds * n_agentic

    print(f"Dataset: {plot_title}")
    print(f"Pool: {pool_desc}")
    print(f"Budget: {args.n_init} init + {args.budget} experiments "
          f"(batch {args.batch_size} -> {rounds} rounds) | "
          f"seeds: {args.seeds} | agent: {args.agent}")
    if args.agent != "heuristic" and n_decisions:
        print(f"Heads-up: up to ~{n_decisions} {args.agent} API calls this run.")
    print()

    def load_cache(fname):
        if args.agent == "heuristic" or args.no_cache:
            return None
        from agentic_bo.cache import PromptCache

        cache_path = os.path.join(".cache", fname)
        cache = PromptCache(cache_path)
        if cache.data:
            print(f"Decision cache: {len(cache.data)} entries at {cache_path} "
                  f"(matching steps resume for free).")
        return cache

    agent = None
    if n_single_agentic:
        agent = build_agent(args.agent, model, args.verbose,
                            cache=load_cache(f"decisions_{args.agent}_{model}.json"))
    tool_agent = None
    if want_tools:
        tool_cache = load_cache(f"tools_{args.worker_model}_{args.reasoner_model}.json")
        tool_agent = build_tool_agent(args.agent, args.worker_model, args.reasoner_model,
                                      args.verbose, cache=tool_cache)
        print(f"agentic_tools routing: worker={args.worker_model} -> "
              f"reasoner={args.reasoner_model}")

    agent_label = args.agent if args.agent == "heuristic" else f"{args.agent}:{model}"
    tools_tag = ""
    if want_tools:  # encode the worker->reasoner routing so configs don't clobber each other
        short = lambda m: m.replace("gemini-2.5-", "").replace("gemini-", "")  # noqa: E731
        tools_tag = (f"_tools_{short(args.worker_model)}-{short(args.reasoner_model)}"
                     if args.agent == "gemini" else "_tools")
    variant = (f"{tools_tag}"
               f"{'_anon' if args.anonymize else ''}{'_permuted' if args.permute_yields else ''}")
    tag = f"{args.dataset}{'_persub' if per_sub else ''}{variant}_{args.agent}"

    traced = False
    if args.langfuse:
        from agentic_bo import tracing

        traced = tracing.enable(session_id=tag, metadata={
            "dataset": args.dataset, "agent": agent_label, "batch_size": args.batch_size,
            "budget": args.budget, "n_init": args.n_init, "seeds": args.seeds,
            "anonymize": args.anonymize, "permute_yields": args.permute_yields,
        })

    factories = {
        "random": lambda: RandomPolicy(),
        "classic_bo": lambda: ClassicBO(),
        "agentic_bo": lambda: AgenticBO(agent, use_surrogate=True),
        "agentic_no_surrogate": lambda: AgenticBO(agent, use_surrogate=False),
        "agentic_tools": lambda: AgenticToolBO(tool_agent),
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

    # Evaluation hygiene: if an agent fell back to EI/heuristic, the "agentic" numbers
    # are not really the model — make that impossible to miss (for both agent flavours).
    for label, a in (("agentic", agent), ("agentic_tools", tool_agent)):
        if a is None or not getattr(a, "calls", 0):
            continue
        hits = getattr(a, "cache_hits", 0)
        cache_note = f" ({hits} tool-calls from cache)" if hits else ""
        fb = a.fallbacks
        if fb:
            print(f"\n[!] {args.agent} {label} fell back on {fb}/{a.calls} decisions "
                  f"({100 * fb / a.calls:.0f}%). Those picks are EI, not the model.{cache_note}")
        else:
            print(f"\n[ok] {args.agent} {label} made all {a.calls} decisions "
                  f"(no fallbacks).{cache_note}")

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

    # Every agentic decision (shortlist + ground truth + the agent's stated strategy and
    # rationale) is logged so the reasoning can be audited against reality afterwards.
    decision_log = []
    for a in (agent, tool_agent):
        decision_log += getattr(a, "decision_log", None) or []
    if decision_log:
        decision_log.sort(key=lambda r: (r["policy"], r["seed"], r["round"]))
        log_path = os.path.join(args.out, f"decisions_{tag}.jsonl")
        with open(log_path, "w") as f:
            for rec in decision_log:
                f.write(json.dumps(rec) + "\n")
        print(f"Saved decision log -> {log_path}")
        print(f"Audit it with: uv run scripts/audit_decisions.py {log_path}")

    if traced:
        from agentic_bo import tracing

        tracing.flush()
        print(f"Langfuse: decisions pushed under session '{tag}'.")


if __name__ == "__main__":
    main()
