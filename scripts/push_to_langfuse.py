"""Backfill an existing decision log into Langfuse — no API calls, no re-run.

Every agentic run writes ``results/decisions_<tag>.jsonl``. This script replays
such a log into Langfuse as one session (= the tag), one trace per policy x seed
campaign, one span per decision round, with the audit scores attached
(pick_percentile, follows_max_ei, overrule_gain_y, batch_best_y, fallback,
cache_hit) — the same definitions as scripts/audit_decisions.py.

Usage:
    uv sync --extra eval
    export LANGFUSE_PUBLIC_KEY=... LANGFUSE_SECRET_KEY=...
    export LANGFUSE_HOST=http://localhost:3000   # self-hosted; omit for cloud
    uv run scripts/push_to_langfuse.py results/decisions_arylation_gemini.jsonl
    uv run scripts/push_to_langfuse.py results/decisions_*.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from agentic_bo import tracing  # noqa: E402


def session_from_path(path: str) -> str:
    """decisions_<tag>.jsonl -> <tag>."""
    name = os.path.splitext(os.path.basename(path))[0]
    return name.removeprefix("decisions_") or name


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("logs", nargs="+", help="decision log(s): results/decisions_<tag>.jsonl")
    p.add_argument("--session-id", default=None,
                   help="override the Langfuse session id (default: the file's tag)")
    args = p.parse_args()

    for path in args.logs:
        with open(path) as f:
            records = [json.loads(line) for line in f if line.strip()]
        session = args.session_id or session_from_path(path)
        if not tracing.enable(session_id=session, metadata={"backfilled_from": path}):
            sys.exit(1)
        for rec in records:
            tracing.log_decision(rec)
        if not tracing.active():  # tracer hard-disables itself on the first error
            sys.exit(f"{path}: push FAILED — see the tracing warning above.")
        tracing.flush()
        campaigns = {(r["policy"], r["seed"]) for r in records}
        print(f"{path}: pushed {len(records)} decisions across {len(campaigns)} "
              f"campaigns to session '{session}'.")


if __name__ == "__main__":
    main()
