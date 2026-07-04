#!/usr/bin/env python3
"""Zero-shot data-leakage probe (LEARNINGS.md next-step 1).

Both reaction datasets are public and plausibly in LLM training data, so the
agent's cold-start advantage could be *recall of this dataset* rather than
chemistry reasoning. This probe asks the model for its best pick with **zero
measurements shown**, many times over seeded random shortlists, and scores the
picks against the ground-truth pool:

* a mean pool-percentile near 50 = no usable prior (a blind pick);
* well above 50 = real prior knowledge — chemistry *or* memorisation;
* run again with ``--anonymize`` (reagent names withheld): a score that falls
  back to ~50 pins the prior on *named* chemistry knowledge (next-step 2),
  while a score that stays high would point at leakage through non-name cues.

The probe is separate from the BO loop on purpose: no surrogate, no history —
whatever it scores comes from the model's prior alone. Decisions are cached in
``.cache/`` like the main loop, so reruns are free.

Usage:
    export GEMINI_API_KEY=...
    uv run scripts/leakage_probe.py --dataset arylation --agent gemini --trials 20
    uv run scripts/leakage_probe.py --dataset arylation --agent gemini --trials 20 --anonymize
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agentic_bo import reactions  # noqa: E402
from agentic_bo.cache import PromptCache  # noqa: E402


def build_probe_prompt(dataset, ids: list) -> str:
    """A zero-shot variant of the loop prompt: legend + shortlist, no measurements."""
    lines = [
        "You are choosing the FIRST experiment of a campaign; no measurements exist yet.",
        f"Objective to maximise: {dataset.objective_label}.",
        "",
        dataset.legend,
        "",
        "Candidate shortlist (no measurements, no surrogate model — use only your own "
        "prior knowledge):",
    ]
    for pos, i in enumerate(ids):
        lines.append(f"  [{pos}] {dataset.descriptions[i]}")
    lines += [
        "",
        "Choose exactly one candidate by its [index], returned as a one-element 'picks' "
        "list: the single most promising experiment according to your prior knowledge.",
    ]
    return "\n".join(lines)


def sample_shortlist(n: int, k: int, seed_pair) -> list:
    rng = np.random.default_rng(seed_pair)
    return [int(i) for i in rng.choice(n, size=k, replace=False)]


def score_pick(dataset, pick_id: int) -> dict:
    y = float(dataset.y[pick_id])
    return {"y": y, "pool_pctile": 100.0 * float(np.mean(dataset.y <= y))}


def build_agent(kind: str, model: str | None):
    if kind == "gemini":
        from agentic_bo.agent import GeminiAgent

        return GeminiAgent(model=model or "gemini-2.5-flash")
    if kind == "claude":
        from agentic_bo.agent import AnthropicAgent

        return AnthropicAgent(model=model or "claude-opus-4-8")
    raise ValueError(f"unknown agent kind: {kind}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=["buchwald", "arylation"], default="arylation")
    p.add_argument("--agent", choices=["gemini", "claude"], default="gemini")
    p.add_argument("--model", default=None, help="LLM model id (defaults per agent)")
    p.add_argument("--trials", type=int, default=20, help="random shortlists to probe")
    p.add_argument("--k", type=int, default=8, help="shortlist size per trial")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--anonymize", action="store_true",
                   help="withhold reagent names (name ablation, LEARNINGS next-step 2)")
    p.add_argument("--out", default="results")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    if args.dataset == "buchwald":
        dataset = reactions.load_buchwald_hartwig(anonymize=args.anonymize)
    else:
        dataset = reactions.load_direct_arylation(anonymize=args.anonymize)

    agent = build_agent(args.agent, args.model)
    cache = PromptCache(os.path.join(".cache", f"probe_{agent.name}_{agent.model}.json"))

    print(f"Zero-shot probe: {dataset.title} | {agent.name}:{agent.model} | "
          f"{args.trials} trials x shortlist {args.k}")
    pool_mean = float(dataset.y.mean())
    print(f"Pool: {dataset.n} candidates | mean yield {pool_mean:.1f} | "
          f"max {dataset.best_value:.1f}\n")

    scores, failures = [], 0
    for t in range(args.trials):
        ids = sample_shortlist(dataset.n, args.k, [args.seed, t])
        prompt = build_probe_prompt(dataset, ids)
        key = cache.key("probe", agent.name, agent.model, prompt)
        raw = cache.get(key)
        if raw is None:
            try:
                raw = agent._raw_decision(prompt)
            except Exception as exc:  # a probe must never silently fall back
                print(f"  trial {t}: API error, skipped ({exc})")
                failures += 1
                continue
            cache.put(key, raw)
        try:
            pos = int(json.loads(raw)["picks"][0])
            pick_id = ids[pos]
        except Exception:
            print(f"  trial {t}: unparseable decision, skipped")
            failures += 1
            continue
        s = score_pick(dataset, pick_id)
        s["best_in_shortlist"] = dataset.y[pick_id] >= max(dataset.y[i] for i in ids)
        scores.append(s)
        if args.verbose:
            print(f"  trial {t}: {dataset.descriptions[pick_id]}"
                  f"  ->  y {s['y']:.1f} ({s['pool_pctile']:.0f}th pool percentile)")

    if not scores:
        raise SystemExit("no successful probe trials")
    if failures:
        print(f"\n[!] {failures}/{args.trials} trials failed and are excluded.")

    pctl = float(np.mean([s["pool_pctile"] for s in scores]))
    mean_y = float(np.mean([s["y"] for s in scores]))
    best_rate = float(np.mean([s["best_in_shortlist"] for s in scores]))
    rand_pctl = 100.0 * float(np.mean([np.mean(dataset.y <= v) for v in dataset.y]))

    print(f"\n=== Zero-shot pick quality over {len(scores)} trials ===")
    print(f"{'':>28} {'model':>8} {'blind pick':>11}")
    print(f"{'mean pool percentile':>28} {pctl:>7.0f}% {rand_pctl:>10.0f}%")
    print(f"{'mean yield of pick':>28} {mean_y:>8.1f} {pool_mean:>11.1f}")
    print(f"{'picked shortlist best':>28} {100 * best_rate:>7.0f}% {100 / args.k:>10.0f}%")
    print("\nReading: ~blind = no usable prior; well above blind = prior knowledge "
          "(chemistry or memorisation). Compare with --anonymize: a drop to blind "
          "pins it on named chemistry knowledge; staying high suggests leakage.")

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(
        args.out,
        f"probe_{args.dataset}{'_anon' if args.anonymize else ''}_{args.agent}.json")
    with open(out_path, "w") as f:
        json.dump({"dataset": dataset.title, "agent": f"{agent.name}:{agent.model}",
                   "trials": len(scores), "k": args.k, "failures": failures,
                   "mean_pool_percentile": pctl, "blind_pool_percentile": rand_pctl,
                   "mean_pick_yield": mean_y, "pool_mean_yield": pool_mean,
                   "best_in_shortlist_rate": best_rate}, f, indent=2)
    print(f"\nSaved probe summary -> {out_path}")


if __name__ == "__main__":
    main()
