"""Tests for the PydanticAI agent: the heuristic FunctionModel drives the *real*
agent loop (tools, validation, retries, structured output) — no API, no network."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

pytest.importorskip("pydantic_ai", reason="uv sync --extra mf")

from pydantic_ai.messages import ModelResponse, RetryPromptPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from datasets import synthetic
from multi_fidelity.agent import INSTRUCTIONS, ROUND_PROMPT, AgenticMF
from multi_fidelity.cache import CachedModel
from multi_fidelity.campaign import run_campaign
from multi_fidelity.heuristic import heuristic_model
from multi_fidelity.oracles import MultiFidelityProblem


def make_problem(rho=0.9, n=150, seed=0):
    return MultiFidelityProblem(synthetic.sample_pool(n=n, seed=0), rho=rho, seed=seed)


def run_heuristic(rho, seed=0, budget=100.0):
    policy = AgenticMF(heuristic_model())
    state = run_campaign(make_problem(rho=rho), policy, seed=seed, budget=budget, n_init=3)
    return state, policy


def test_heuristic_agent_completes_a_campaign():
    state, policy = run_heuristic(rho=0.9)
    assert state.ledger.spent <= 100.0 + 1e-9
    assert len(state.hf_obs) > 3                       # it measured beyond the init
    assert policy.decision_log                         # every round logged a decision
    for rec in policy.decision_log:
        assert rec["rationale"] and rec["spent_after"] >= rec["spent_before"]
    assert policy.decision_log[-1]["stop"] or not state.can_measure()


def test_heuristic_agent_adapts_its_trust_to_the_simulator():
    """The whole point of the experiment: more simulation when the simulator is
    good (rho high), near-none once the LF/HF pairs expose it as junk."""
    sims = {}
    for rho in (0.95, 0.0):
        state, _ = run_heuristic(rho=rho, seed=0)
        sims[rho] = len(state.lf_obs)
    assert sims[0.95] > sims[0.0]


def test_invalid_tool_calls_are_retried_not_charged():
    """A bad call (re-measuring a known candidate) must cost nothing: ModelRetry
    bounces it back to the model, which then corrects itself."""
    prob = make_problem()
    # run_campaign's init is seeded, so the already-measured ids are known up front
    init_id = int(np.random.default_rng(0).choice(prob.n, size=2, replace=False)[0])
    saw_retry = {"flag": False}

    def scripted(messages, info):
        for part in messages[-1].parts:
            if isinstance(part, RetryPromptPart):
                saw_retry["flag"] = True
                return ModelResponse(parts=[ToolCallPart(
                    info.output_tools[0].name,
                    {"rationale": "corrected after retry", "stop": True})])
        return ModelResponse(parts=[ToolCallPart("measure", {"candidate_ids": [init_id]})])

    policy = AgenticMF(FunctionModel(scripted, model_name="scripted"))
    state = run_campaign(prob, policy, seed=0, budget=60.0, n_init=2)
    assert saw_retry["flag"]                           # the invalid call was bounced
    assert state.ledger.spent == 2 * prob.cost_hf      # ...and never charged
    assert policy.decision_log[-1]["rationale"] == "corrected after retry"


def test_heuristic_campaign_trajectory_is_pinned():
    """Golden trajectory: the exact (candidate, fidelity) sequence of one short
    campaign. Property tests pass for any valid behaviour; this test fails on ANY
    behaviour change — a heuristic edit, a tie-break shift, an sklearn upgrade
    that moves the GP fit. Changing behaviour on purpose means updating this list
    on purpose, like the golden prompt test."""
    prob = MultiFidelityProblem(synthetic.sample_pool(n=60, seed=0), rho=0.9, seed=0)
    policy = AgenticMF(heuristic_model())
    state = run_campaign(prob, policy, seed=0, budget=60.0, n_init=2)
    assert [(o.candidate, o.fidelity) for o in state.ledger.log] == [
        (50, "hf"), (38, "hf"),                       # seeded init
        (33, "lf"), (7, "lf"), (24, "lf"), (7, "hf"),     # round 1: screen 3, confirm best
        (46, "lf"), (27, "lf"), (31, "lf"), (46, "hf"),   # round 2
        (15, "lf"), (20, "lf"), (26, "lf"), (15, "hf"),   # round 3, then stop (1 unit left)
    ]
    assert policy.decision_log[-1]["stop"] is True


def test_llm_failure_falls_back_instead_of_crashing():
    """An exhausted-retries round (here: a model that keeps calling an invalid
    id) is logged as a fallback, spends nothing, and the idle guard ends the
    campaign — instead of an UnexpectedModelBehavior killing the whole sweep."""

    def broken(messages, info):
        return ModelResponse(parts=[ToolCallPart("measure", {"candidate_ids": [999_999]})])

    policy = AgenticMF(FunctionModel(broken, model_name="broken"))
    prob = make_problem()
    state = run_campaign(prob, policy, seed=0, budget=60.0, n_init=2)
    assert state.ledger.spent == 2 * prob.cost_hf          # nothing beyond the init
    assert policy.decision_log and all(r["fallback"] for r in policy.decision_log)
    assert "UnexpectedModelBehavior" in policy.decision_log[-1]["error"]
    assert len(policy.decision_log) == 2                   # idle guard: two dry rounds, done


def test_rounds_log_requests_and_retries():
    _, policy = run_heuristic(rho=0.9, budget=60.0)
    for rec in policy.decision_log:
        assert rec["n_requests"] >= 3          # recall + shortlist + at least one action
        assert rec["n_retries"] == 0 and not rec["fallback"]
        assert rec["cost_hf"] == 10.0


def test_cached_model_replays_a_campaign_for_free(tmp_path):
    path = tmp_path / "cache.json"
    runs = []
    for _ in range(2):
        model = CachedModel(heuristic_model(), path)
        state, _ = run_and_capture(model)
        runs.append((state.ledger.log, model.hits, model.misses))
    (log1, _, misses1), (log2, hits2, misses2) = runs
    assert misses1 > 0 and misses2 == 0 and hits2 == misses1   # second run fully cached
    assert log1 == log2                                        # identical campaign


def run_and_capture(model):
    policy = AgenticMF(model)
    state = run_campaign(make_problem(), policy, seed=0, budget=80.0, n_init=3)
    return state, policy


def test_recall_parsing_is_section_aware():
    """Regression: the simulator-vs-truth pairs share the ``[id=..] sim=..`` line
    shape with the unconfirmed list. A pool-wide regex once read the (already
    measured) pairs as unconfirmed candidates, so the heuristic tried to
    re-measure one and burned all its retries on the same deterministic call."""
    from multi_fidelity.heuristic import _parse_recall

    recall = (
        "Budget: remaining=57 spent=143 cost_lf=1 cost_hf=10\n"
        "Measured (3):\n"
        "  [id=4] cand-4 -> 61.00\n"
        "  [id=9] cand-9 -> 40.00\n"
        "  [id=2] cand-2 -> 12.00\n"
        "Simulator vs truth on 1 shared candidate(s):\n"
        "  [id=4] sim=99.00 true=61.00 error=+38.00\n"
        "Simulated only (2), top by simulated value:\n"
        "  [id=7] sim=55.00\n"
        "  [id=3] sim=20.00\n"
    )
    parsed = _parse_recall(recall)
    assert parsed["budget"] == (57.0, 1.0, 10.0)
    assert parsed["measured_vals"] == [61.0, 40.0, 12.0]
    assert parsed["errors"] == [38.0]
    # id=4 (measured, sim=99) must NOT appear as unconfirmed
    assert parsed["unconfirmed"] == [(7, 55.0), (3, 20.0)]


def test_heuristic_survives_many_seeds():
    """The retry crash above only surfaced at seed counts the smoke runs missed;
    sweep a batch of seeds through full campaigns."""
    prob = make_problem(rho=0.9, n=120)
    for seed in range(10):
        policy = AgenticMF(heuristic_model())
        state = run_campaign(prob, policy, seed=seed, budget=90.0, n_init=3)
        assert state.ledger.spent <= 90.0 + 1e-9


def test_audit_script_reports_on_a_decision_log(tmp_path):
    import json
    import subprocess
    import sys as _sys

    _, policy = run_heuristic(rho=0.9, budget=80.0)
    log_path = tmp_path / "decisions.jsonl"
    log_path.write_text("\n".join(json.dumps({"seed": 0, **r}) for r in policy.decision_log))
    repo = os.path.join(os.path.dirname(__file__), "..", "..")
    out = subprocess.run(
        [_sys.executable, os.path.join(repo, "scripts", "audit_mf_decisions.py"),
         str(log_path), "--show", "2"],
        capture_output=True, text=True, check=True).stdout
    for section in ("Fidelity mix", "Screen -> confirm", "Loop health", "Sample rationales"):
        assert section in out
    assert "fallback rounds: 0/" in out


def test_prompt_format_is_stable():
    """The replay cache is keyed by the exact request; pin the wording so an
    accidental edit cannot silently turn cache hits into paid API calls."""
    assert INSTRUCTIONS.startswith("You run a budgeted experimental campaign")
    assert "reliability is UNKNOWN" in INSTRUCTIONS
    rendered = ROUND_PROMPT.format(round=2, spent=40.0, budget=200.0, remaining=160.0,
                                   cost_lf=1.0, cost_hf=10.0, best=61.25, memory="(none)")
    assert rendered == (
        "Round 2. Spent 40 of 200; remaining 160. "
        "Costs per candidate: simulate 1, measure 10.\n"
        "Best measured so far: 61.25.\n"
        "Notes from earlier rounds:\n(none)\n\n"
        "Investigate with recall/shortlist/predict, then spend budget with simulate/measure. "
        "Finish with your rationale, a short note to your future self, and stop=true only "
        "if further spending cannot improve the best measurement."
    )
