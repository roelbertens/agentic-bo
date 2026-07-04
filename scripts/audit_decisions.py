#!/usr/bin/env python3
"""Audit the agent's decisions against ground truth (LEARNINGS.md next-step 3).

Every agentic run writes ``results/decisions_<tag>.jsonl``: one record per decision
with the full shortlist (descriptions, true yields, surrogate stats) next to the
agent's stated ``strategy`` and ``rationale``. This script checks that the stated
reasoning is connected to reality — after the fact, with zero API calls:

* **Strategy vs outcome** — do "exploit" rounds actually yield more than "explore"
  rounds, and does the stated strategy match what was picked (exploit -> high
  predicted mean, explore -> high uncertainty)?
* **Surrogate agreement** — how often does the agent just follow max-EI, and when
  it overrules the surrogate, does that pay off (yield gained vs the EI pick)?
* **Leakage indicator** — on ``--permute-yields`` runs the record keeps both the
  observed (permuted) and the original yields. A pick quality that stays high
  under the *original* yields, which the agent was never shown, means the model
  recalls the dataset rather than reasoning (LEARNINGS.md next-step 1).

Usage:
    uv run scripts/audit_decisions.py results/decisions_arylation_gemini.jsonl
    uv run scripts/audit_decisions.py results/decisions_*.jsonl --show 5
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np


def load_records(paths: list) -> list:
    records = []
    for path in paths:
        with open(path) as f:
            records.extend(json.loads(line) for line in f if line.strip())
    return records


def _percentile_in_shortlist(record: dict, key: str = "y") -> float:
    """Percentile (0-100) of the top pick's yield within its own shortlist."""
    ys = np.array([c[key] for c in record["shortlist"]])
    pick_y = record["shortlist"][record["picked_pos"][0]][key]
    return 100.0 * float(np.mean(ys <= pick_y))


def _top_pick(record: dict) -> dict:
    return record["shortlist"][record["picked_pos"][0]]


def _batch_best(record: dict, key: str = "y") -> float:
    return max(record["shortlist"][p][key] for p in record["picked_pos"])


def audit_strategy_vs_outcome(records: list) -> None:
    print("\n== Strategy vs outcome (is the stated strategy connected to reality?) ==")
    header = (f"  {'strategy':<10} {'n':>5} {'top-pick y':>11} {'batch best':>11} "
              f"{'sl best':>8} {'pick %ile':>10} {'mean round':>11}")
    print(header)
    by_strategy = defaultdict(list)
    for r in records:
        by_strategy[r.get("strategy") or "?"].append(r)
    for strat, recs in sorted(by_strategy.items(), key=lambda kv: -len(kv[1])):
        top_y = np.mean([_top_pick(r)["y"] for r in recs])
        batch = np.mean([_batch_best(r) for r in recs])
        sl_best = np.mean([max(c["y"] for c in r["shortlist"]) for r in recs])
        pctl = np.mean([_percentile_in_shortlist(r) for r in recs])
        rnd = np.mean([r["round"] for r in recs])
        print(f"  {strat:<10} {len(recs):>5} {top_y:>11.1f} {batch:>11.1f} "
              f"{sl_best:>8.1f} {pctl:>9.0f}% {rnd:>11.1f}")
    print("  (pick %ile = top pick's yield percentile within its shortlist; "
          "50% would be a blind pick)")

    # Does the pick match the stated strategy, judged by the surrogate stats it saw?
    with_surr = [r for r in records if "ei" in r["shortlist"][0]]
    if with_surr:
        rows = []
        for strat, stat_key in [("exploit", "mean"), ("explore", "std")]:
            recs = [r for r in with_surr if r.get("strategy") == strat]
            if not recs:
                continue
            hits = 0
            for r in recs:
                vals = np.array([c[stat_key] for c in r["shortlist"]])
                top3 = set(np.argsort(-vals)[:3])
                hits += r["picked_pos"][0] in top3
            rows.append(f"  '{strat}' picks in the shortlist's top-3 by {stat_key}: "
                        f"{hits}/{len(recs)} ({100 * hits / len(recs):.0f}%)")
        if rows:
            print("\n  Consistency of stated strategy with what was picked:")
            for row in rows:
                print(row)


def audit_surrogate_agreement(records: list) -> None:
    with_surr = [r for r in records if "ei" in r["shortlist"][0]]
    if not with_surr:
        return
    print("\n== Surrogate agreement (does the agent add anything over max-EI?) ==")
    followed, overrules = [], []
    for r in with_surr:
        ei = np.array([c["ei"] for c in r["shortlist"]])
        ei_pos = int(np.argmax(ei))
        if r["picked_pos"][0] == ei_pos:
            followed.append(r)
        else:
            overrules.append((r, ei_pos))
    n = len(with_surr)
    print(f"  followed max-EI: {len(followed)}/{n} ({100 * len(followed) / n:.0f}%)")
    if overrules:
        gain = [_top_pick(r)["y"] - r["shortlist"][ei_pos]["y"] for r, ei_pos in overrules]
        wins = sum(g > 0 for g in gain)
        print(f"  overruled max-EI: {len(overrules)}/{n} — mean yield gain vs the EI pick: "
              f"{np.mean(gain):+.1f} (won {wins}/{len(overrules)})")
        print("  (a consistently negative gain means the agent's overrides hurt; "
              "positive means its chemistry adds real signal)")


def audit_leakage(records: list) -> None:
    permuted = [r for r in records if "y_true" in r["shortlist"][0]]
    if not permuted:
        return
    print("\n== Leakage indicator (permuted-yields run) ==")
    obs = np.mean([_percentile_in_shortlist(r, "y") for r in permuted])
    true = np.mean([_percentile_in_shortlist(r, "y_true") for r in permuted])
    print(f"  top-pick percentile under OBSERVED (permuted) yields: {obs:.0f}%")
    print(f"  top-pick percentile under ORIGINAL yields (never shown): {true:.0f}%")
    print("  A blind pick sits at ~50% for both. Tracking the observed yields is honest "
          "in-context learning; sitting well above 50% on the ORIGINAL yields means the "
          "model recalls the dataset (leakage), since chemistry no longer maps to reward.")
    by_round = defaultdict(list)
    for r in permuted:
        by_round[r["round"]].append(_percentile_in_shortlist(r, "y_true"))
    trail = "  original-yield %ile by round: " + "  ".join(
        f"r{k}:{np.mean(v):.0f}" for k, v in sorted(by_round.items()))
    print(trail)


def show_rationales(records: list, n: int) -> None:
    print(f"\n== Sample rationales (first {n}; check the chemistry claims by hand) ==")
    for r in records[:n]:
        pick = _top_pick(r)
        sl_best = max(c["y"] for c in r["shortlist"])
        print(f"\n  [{r['policy']} seed {r['seed']} round {r['round']}] "
              f"strategy={r.get('strategy')}"
              + (" (FALLBACK — not the model)" if r.get("fallback") else ""))
        print(f"    rationale: {r.get('rationale')}")
        print(f"    picked: {pick['desc']}")
        print(f"    -> yield {pick['y']:.1f} (shortlist best was {sl_best:.1f})")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("logs", nargs="+", help="decision log(s): results/decisions_<tag>.jsonl")
    p.add_argument("--policy", default=None,
                   help="only audit this policy (agentic_bo / agentic_no_surrogate)")
    p.add_argument("--show", type=int, default=0, metavar="N",
                   help="also print the first N rationales with their outcomes")
    args = p.parse_args()

    records = load_records(args.logs)
    if args.policy:
        records = [r for r in records if r["policy"] == args.policy]
    if not records:
        raise SystemExit("no decision records found")

    datasets = sorted({r["dataset"] for r in records})
    policies = sorted({r["policy"] for r in records})
    fallbacks = sum(r.get("fallback", False) for r in records)
    print(f"Decisions: {len(records)} | datasets: {', '.join(datasets)} | "
          f"policies: {', '.join(policies)}")
    if fallbacks:
        print(f"[!] {fallbacks} decisions were fallbacks (EI/heuristic, not the model) — "
              f"they are excluded from the reasoning audits below.")
    model_records = [r for r in records if not r.get("fallback")]

    for policy in policies:
        recs = [r for r in model_records if r["policy"] == policy]
        if not recs:
            continue
        print(f"\n#### {policy} ({len(recs)} model decisions) " + "#" * 20)
        audit_strategy_vs_outcome(recs)
        audit_surrogate_agreement(recs)
        audit_leakage(recs)
        if args.show:
            show_rationales(recs, args.show)


if __name__ == "__main__":
    main()
