"""Offline tests for the genuinely-agentic tool policy (LEARNINGS next-step 11).

All run through the deterministic HeuristicToolAgent, so they exercise the tool
surface, the cross-round memory and the decision logging without any API calls."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agentic_bo import objective
from agentic_bo.agentic import BOEnvironment, HeuristicToolAgent
from agentic_bo.data import permute_yields
from agentic_bo.experiment import run_method, run_single
from agentic_bo.policies import AgenticBO, AgenticToolBO
from agentic_bo.surrogate import GaussianSurrogate, expected_improvement


def _env(ds, evaluated, y_obs, memory="", seed=0):
    unseen = np.array([i for i in range(ds.n) if i not in set(evaluated)])
    surrogate = GaussianSurrogate(seed=seed).fit(ds.X[evaluated], np.asarray(y_obs))
    m, s = surrogate.predict(ds.X[unseen])
    ei = expected_improvement(m, s, best=float(max(y_obs)))
    return BOEnvironment(ds, unseen, m, s, ei, memory, np.random.default_rng(seed))


def test_environment_tools_return_valid_candidates():
    ds = objective.sample_pool(n=120, seed=0)
    evaluated, y_obs = [0, 1, 2, 3], list(ds.y[[0, 1, 2, 3]])
    env = _env(ds, evaluated, y_obs)

    top = env.search_candidates("top_ei", limit=5)
    assert len(top) == 5 and all(env.valid_id(c["id"]) for c in top)
    assert all(c["id"] not in evaluated for c in top)          # never returns measured points

    preds = env.predict([c["id"] for c in top])
    assert {"id", "desc", "mean", "std", "ei"} <= set(preds[0])
    # top_ei really is sorted by EI
    eis = [p["ei"] for p in env.predict([c["id"] for c in top])]
    assert eis == sorted(eis, reverse=True)

    assert env.predict([999999]) == []                          # out-of-range ids are dropped
    assert "empty" in env.recall()["memory"]                    # no memory yet


def test_search_match_filters_descriptions():
    ds = objective.sample_pool(n=200, seed=0)
    env = _env(ds, [0, 1, 2], list(ds.y[[0, 1, 2]]))
    hits = env.search_candidates("match", match=ds.descriptions[50].split(",")[0], limit=5)
    assert hits and all(ds.descriptions[50].split(",")[0] in ds.descriptions[c["id"]] for c in hits)


def test_heuristic_tool_agent_drives_a_round():
    ds = objective.sample_pool(n=150, seed=0)
    env = _env(ds, [0, 1, 2, 3], list(ds.y[[0, 1, 2, 3]]))
    dec = HeuristicToolAgent().propose_round(env, "round 1 ... remaining ...", q=3)
    assert len(dec.picks) == 3 and len(set(dec.picks)) == 3
    assert all(env.valid_id(i) for i in dec.picks)
    assert dec.tool_calls and dec.note                          # it used tools + wrote a note
    assert {t["name"] for t in dec.tool_calls} >= {"search_candidates", "predict"}


def test_memory_persists_across_rounds():
    ds = objective.sample_pool(n=150, seed=0)
    policy = AgenticToolBO(HeuristicToolAgent())
    assert policy.memory == ""
    run_single(ds, policy, seed=0, budget=6, n_init=3, batch_size=2)
    assert policy.memory.count("R") >= 2                        # a note was carried each round
    assert policy.memory.startswith("R")


def test_tool_policy_curve_is_valid_and_logs():
    ds = objective.sample_pool(n=150, seed=0)
    agent = HeuristicToolAgent()
    policy = AgenticToolBO(agent)
    curve = run_single(ds, policy, seed=1, budget=8, n_init=4, batch_size=2)
    assert np.all(np.diff(curve) >= 0) and curve[-1] <= ds.best_value + 1e-9
    assert len(agent.decision_log) == 4                         # ceil(8/2) rounds
    rec = agent.decision_log[0]
    assert rec["policy"] == "agentic_tools" and rec["n_tool_calls"] > 0
    assert rec["picked_pos"] and all(0 <= p < len(rec["shortlist"]) for p in rec["picked_pos"])
    assert "phase_models" in rec and rec["n_considered"] > 0


def test_permuted_run_keeps_y_true_in_the_tool_log():
    ds = permute_yields(objective.sample_pool(n=120, seed=0), seed=1)
    agent = HeuristicToolAgent()
    run_single(ds, AgenticToolBO(agent), seed=0, budget=4, n_init=3, batch_size=2)
    for entry in agent.decision_log[0]["shortlist"]:
        assert entry["y"] == float(ds.y[entry["id"]])
        assert entry["y_true"] == float(ds.y_true[entry["id"]])


def test_tool_agent_is_at_least_competitive_offline():
    """Sanity floor: the tool loop shouldn't do worse than the single-call heuristic
    agent on the smooth synthetic task (same deterministic UCB decision underneath)."""
    from agentic_bo.agent import HeuristicAgent

    ds = objective.sample_pool(n=400, seed=0)
    seeds = range(6)
    single = run_method(ds, lambda: AgenticBO(HeuristicAgent()), "agentic_bo", seeds,
                        15, 5, verbose=False)
    tools = run_method(ds, lambda: AgenticToolBO(HeuristicToolAgent()), "agentic_tools", seeds,
                       15, 5, verbose=False)
    assert tools.curves[:, -1].mean() >= single.curves[:, -1].mean() - 2.0
