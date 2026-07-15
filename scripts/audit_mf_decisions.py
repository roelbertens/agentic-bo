#!/usr/bin/env python3
"""Audit a multi-fidelity agent's decision log against its behaviour.

Reads a ``results/multi_fidelity/decisions_mf_*.jsonl`` written by ``run_multi_fidelity.py``
(one record per agent round, per seed) and reports, without any API calls,
whether the agent's stated reasoning matches what it actually did:

* the fidelity mix — how spend splits between simulations and measurements,
  early versus late in the campaign;
* screen-then-confirm behaviour — how many measurements were simulated first,
  and whether confirmed picks out-yield direct ones;
* the simulator accuracy the agent itself observed (candidates with both a
  simulated and a measured value);
* loop health — fallback rounds, requests and retries per round, and whether
  stops were voluntary (budget left) or forced (budget dry).

Usage:
    uv run scripts/audit_mf_decisions.py results/multi_fidelity/decisions_mf_<tag>.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict


def load(path: str) -> dict[int, list[dict]]:
    per_seed: dict[int, list[dict]] = defaultdict(list)
    with open(path) as fh:
        for line in fh:
            rec = json.loads(line)
            per_seed[rec["seed"]].append(rec)
    for rounds in per_seed.values():
        rounds.sort(key=lambda r: r["round"])
    return dict(per_seed)


def mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def audit(per_seed: dict[int, list[dict]], show: int) -> None:
    n_seeds = len(per_seed)
    rounds_all = [r for rounds in per_seed.values() for r in rounds]
    first = rounds_all[0]
    cost_lf, cost_hf = first["cost_lf"], first["cost_hf"]

    print(f"Campaigns: {n_seeds} seeds, {mean(len(r) for r in per_seed.values()):.1f} "
          f"rounds each; costs simulate {cost_lf:g} / measure {cost_hf:g}, "
          f"budget {first.get('budget', '?')}")

    # --- fidelity mix -------------------------------------------------------------
    n_sim = sum(len(r["simulated"]) for r in rounds_all)
    n_meas = sum(len(r["measured"]) for r in rounds_all)
    spend_sim, spend_meas = n_sim * cost_lf, n_meas * cost_hf
    total = spend_sim + spend_meas
    print("\nFidelity mix")
    print(f"  {n_sim} simulations ({spend_sim:g} spend) vs {n_meas} measurements "
          f"({spend_meas:g} spend) -> {100 * spend_meas / total:.0f}% of agent spend on HF")
    halves = {"early": [], "late": []}
    for rounds in per_seed.values():
        mid = len(rounds) / 2
        for r in rounds:
            key = "early" if r["round"] <= mid else "late"
            halves[key].append((len(r["simulated"]), len(r["measured"])))
    for key, pairs in halves.items():
        s, m = sum(p[0] for p in pairs), sum(p[1] for p in pairs)
        hf_share = m * cost_hf / max(s * cost_lf + m * cost_hf, 1e-9)
        print(f"  {key:5s} rounds: {s} sim / {m} meas ({100 * hf_share:.0f}% HF spend)")

    # --- screen -> confirm --------------------------------------------------------
    confirmed_vals, direct_vals = [], []
    sim_err = []
    for rounds in per_seed.values():
        seen_sims: dict[str, float] = {}
        for r in rounds:
            seen_sims.update(r["simulated"])
            for cand, value in r["measured"].items():
                (confirmed_vals if cand in seen_sims else direct_vals).append(value)
                if cand in seen_sims:
                    sim_err.append(abs(seen_sims[cand] - value))
    print("\nScreen -> confirm")
    n_conf, n_dir = len(confirmed_vals), len(direct_vals)
    print(f"  {n_conf}/{n_conf + n_dir} measurements were simulated first")
    if confirmed_vals and direct_vals:
        print(f"  mean measured value: confirmed {mean(confirmed_vals):.1f} "
              f"vs direct {mean(direct_vals):.1f}")
    if sim_err:
        print(f"  simulator error the agent observed: mean |sim - true| = {mean(sim_err):.1f}")

    # --- loop health ----------------------------------------------------------------
    fallbacks = [r for r in rounds_all if r.get("fallback")]
    retries = [r for r in rounds_all if (r.get("n_retries") or 0) > 0]
    print("\nLoop health")
    print(f"  fallback rounds: {len(fallbacks)}/{len(rounds_all)}"
          + (f" (first error: {fallbacks[0]['error']})" if fallbacks else ""))
    req = mean(r["n_requests"] for r in rounds_all if r["n_requests"])
    print(f"  requests per round: mean {req:.1f}; rounds with retries: {len(retries)}")
    stopped, dry = 0, 0
    for rounds in per_seed.values():
        last = rounds[-1]
        remaining = last.get("budget", 0) - last["spent_after"]
        if last["stop"] and remaining >= cost_hf:
            stopped += 1                     # voluntary: money left for a measurement
        else:
            dry += 1
    print(f"  campaign endings: {dry} budget-dry, {stopped} voluntary stop with >= 1 "
          f"measurement affordable")

    # --- sample rationales ----------------------------------------------------------
    if show:
        print("\nSample rationales")
        for r in rounds_all[:show]:
            spent = r["spent_after"] - r["spent_before"]
            print(f"  [seed {r['seed']} round {r['round']}] spent {spent:g}, "
                  f"{len(r['simulated'])} sim / {len(r['measured'])} meas: {r['rationale']}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("log", help="results/multi_fidelity/decisions_mf_<tag>.jsonl")
    p.add_argument("--show", type=int, default=0, help="print the first N rationales")
    args = p.parse_args()
    audit(load(args.log), args.show)


if __name__ == "__main__":
    main()
