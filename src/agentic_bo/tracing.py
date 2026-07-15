"""Optional Langfuse tracing: push agentic decisions to a trace/score store.

Design goals, in order:

* **Zero footprint by default.** Without ``enable()`` every call here is a no-op;
  the package imports nothing beyond the stdlib. CI and offline runs are untouched.
* **Soft-fail.** If ``langfuse`` is missing, keys are unset, or the server is
  down, tracing warns once and disables itself — it must never crash a run.
* **Same numbers as the audit.** The per-decision scores pushed to Langfuse are
  computed by ``derived_scores`` with the exact definitions used by
  ``scripts/audit_decisions.py`` (pick percentile, follows-max-EI, overrule gain),
  so the UI and the offline audit can never disagree.

Mapping onto Langfuse concepts:

* session  = one ``run_agentic_bo.py`` invocation (the results tag, e.g. ``arylation_gemini``)
* trace    = one campaign (policy x seed), e.g. ``agentic_bo seed 3``
* span     = one decision round (input: shortlist + surrogate stats;
             output: picks, strategy, rationale)
* scores   = per-round quality/behaviour metrics attached to the round span

Enable with ``run_agentic_bo.py --langfuse`` (live) or backfill any existing decision log
with ``scripts/push_to_langfuse.py``. Requires ``uv sync --extra eval`` and
LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY (+ LANGFUSE_HOST for self-hosted).
"""
from __future__ import annotations

import os

_client = None            # Langfuse client once enabled; None = no-op
_session: str = ""
_run_metadata: dict = {}
_trace_named: set = set()  # (policy, seed) campaigns whose trace attrs are set
_failed = False


def derived_scores(record: dict) -> dict:
    """Per-decision scores, matching scripts/audit_decisions.py definitions.

    Pure and dependency-free so it is unit-testable without langfuse installed.
    """
    shortlist = record.get("shortlist") or []
    picked = record.get("picked_pos") or []
    scores: dict[str, float] = {}
    if shortlist and picked and all(0 <= p < len(shortlist) for p in picked):
        ys = [c["y"] for c in shortlist]
        top_y = shortlist[picked[0]]["y"]
        scores["batch_best_y"] = max(shortlist[p]["y"] for p in picked)
        scores["pick_percentile"] = 100.0 * sum(y <= top_y for y in ys) / len(ys)
        if all(c.get("ei") is not None for c in shortlist):
            ei_pos = max(range(len(shortlist)), key=lambda i: shortlist[i]["ei"])
            follows = picked[0] == ei_pos
            scores["follows_max_ei"] = float(follows)
            if not follows:
                scores["overrule_gain_y"] = top_y - shortlist[ei_pos]["y"]
    scores["fallback"] = float(bool(record.get("fallback")))
    scores["cache_hit"] = float(bool(record.get("cache_hit")))
    return scores


def enable(session_id: str, metadata: dict | None = None) -> bool:
    """Turn tracing on for this process. Returns False (and warns) if unavailable."""
    global _client, _session, _run_metadata
    try:
        from langfuse import Langfuse
    except ImportError:
        print("[warn] tracing: langfuse is not installed — run `uv sync --extra eval`. "
              "Tracing disabled.")
        return False
    if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
        print("[warn] tracing: set LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY "
              "(and LANGFUSE_HOST for self-hosted). Tracing disabled.")
        return False
    _client = Langfuse()
    _session = session_id
    _run_metadata = metadata or {}
    _trace_named.clear()
    return True


def log_decision(record: dict) -> None:
    """Push one decision-log record as a round span + scores. No-op unless enabled."""
    global _client, _failed
    if _client is None:
        return
    try:
        from langfuse import Langfuse

        campaign = (record["policy"], record["seed"])
        trace_id = Langfuse.create_trace_id(
            seed=f"{_session}:{record['policy']}:{record['seed']}")
        span_kwargs = dict(
            trace_context={"trace_id": trace_id},
            name=f"round {record['round']:02d}",
            input={
                "round": record["round"], "n_rounds": record.get("n_rounds"),
                "best_so_far": record.get("best_so_far"),
                "shortlist": record.get("shortlist"),
            },
            output={
                "picked_pos": record.get("picked_pos"),
                "strategy": record.get("strategy"),
                "rationale": record.get("rationale"),
            },
            metadata={k: record[k] for k in
                      ("fallback", "cache_hit", "latency_s", "n_tool_calls")
                      if record.get(k) is not None},
        )
        if hasattr(_client, "start_observation"):  # SDK v4: OTel-native
            from langfuse import propagate_attributes

            with propagate_attributes(
                    session_id=_session,
                    trace_name=f"{record['policy']} seed {record['seed']}",
                    tags=[record["policy"], record["dataset"]],
                    metadata={k: str(v) for k, v in _run_metadata.items()}):
                span = _client.start_observation(**span_kwargs)
        else:  # SDK v3
            span = _client.start_span(**span_kwargs)
            if campaign not in _trace_named:
                _trace_named.add(campaign)
                span.update_trace(
                    name=f"{record['policy']} seed {record['seed']}",
                    session_id=_session,
                    tags=[record["policy"], record["dataset"]],
                    metadata=_run_metadata,
                )
        for name, value in derived_scores(record).items():
            span.score(name=name, value=value)
        span.end()
    except Exception as exc:  # tracing must never take down a run
        if not _failed:
            print(f"[warn] tracing: disabled after error: {exc}")
            _failed = True
        _client = None


def active() -> bool:
    """True while tracing is enabled and has not hard-disabled after an error."""
    return _client is not None


def flush() -> None:
    """Block until queued events are delivered (call once, at the end of a run)."""
    if _client is not None:
        _client.flush()
