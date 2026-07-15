"""Tracing must be a safe no-op offline, and its scores must match the audit's."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import numpy as np
import pytest

from agentic_bo import tracing

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
from audit_decisions import _batch_best, _percentile_in_shortlist  # noqa: E402


def _record(picked=(1,), ei=(0.1, 0.5, 0.2), y=(30.0, 55.0, 42.0)):
    return {
        "dataset": "t", "policy": "agentic_bo", "seed": 0, "round": 1, "n_rounds": 5,
        "best_so_far": 40.0, "n_select": len(picked), "strategy": "exploit",
        "rationale": "r", "fallback": False, "cache_hit": True, "latency_s": None,
        "picked_pos": list(picked),
        "shortlist": [{"id": i, "desc": f"c{i}", "y": y[i], "ei": ei[i]}
                      for i in range(len(y))],
    }


def test_log_decision_is_a_noop_when_not_enabled():
    tracing.log_decision(_record())  # must not raise, must not need langfuse


def test_enable_fails_soft_without_langfuse_or_keys(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    assert tracing.enable("test-session") is False


def test_derived_scores_match_the_audit_definitions():
    rec = _record(picked=(2, 0))  # top pick overrules max-EI (pos 1)
    s = tracing.derived_scores(rec)
    assert s["batch_best_y"] == _batch_best(rec)
    assert s["pick_percentile"] == pytest.approx(_percentile_in_shortlist(rec))
    assert s["follows_max_ei"] == 0.0
    assert s["overrule_gain_y"] == rec["shortlist"][2]["y"] - rec["shortlist"][1]["y"]
    assert s["cache_hit"] == 1.0 and s["fallback"] == 0.0


def test_derived_scores_when_the_agent_follows_max_ei():
    s = tracing.derived_scores(_record(picked=(1,)))
    assert s["follows_max_ei"] == 1.0
    assert "overrule_gain_y" not in s
    assert s["pick_percentile"] == 100.0  # pos 1 is also the best yield


def test_derived_scores_without_surrogate_stats():
    rec = _record()
    for c in rec["shortlist"]:
        c["ei"] = None
    s = tracing.derived_scores(rec)
    assert "follows_max_ei" not in s and "batch_best_y" in s


def test_llm_agent_records_latency_field(tmp_path):
    """The decision log must carry latency_s (None when served from cache/heuristic)."""
    from agentic_bo.agent import DecisionContext, _LLMAgent

    class Fake(_LLMAgent):
        name, model = "fake", "m"

        def _raw_decision(self, prompt):
            return '{"picks": [0], "strategy": "exploit", "rationale": "x"}'

    ctx = DecisionContext(
        iteration=1, budget=5, remaining=5, objective_label="y", legend="",
        hist_desc=["a"], y_eval=np.array([1.0]), X_eval=np.zeros((1, 2)),
        cand_desc=["b", "c"], cand_ids=[0, 1], X_cand=np.zeros((2, 2)),
        mean=None, std=None, ei=None, use_surrogate=False, n_select=1,
    )
    agent = Fake()
    agent.select_batch(ctx)
    assert agent.last_decision["latency_s"] is not None
    assert agent.last_decision["latency_s"] >= 0.0
