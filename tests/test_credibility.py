"""Tests for the credibility-check tooling: permuted yields, name ablation,
decision logging + rationale audit, and the zero-shot probe helpers (all offline)."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from agentic_bo import objective, reactions
from agentic_bo.agent import DecisionContext, HeuristicAgent, _build_prompt
from agentic_bo.data import permute_yields
from agentic_bo.experiment import run_single
from agentic_bo.policies import AgenticBO

REPO = os.path.join(os.path.dirname(__file__), "..")
BUCHWALD = os.path.join(REPO, "data", "buchwald_hartwig.xlsx")
ARYLATION = os.path.join(REPO, "data", "direct_arylation", "experiment_index.csv")

needs_buchwald = pytest.mark.skipif(
    not os.path.exists(BUCHWALD), reason="run scripts/get_data.py")
needs_arylation = pytest.mark.skipif(
    not os.path.exists(ARYLATION), reason="run scripts/get_data.py")


def test_permute_yields_is_seeded_and_keeps_the_pool():
    ds = objective.sample_pool(n=50, seed=0)
    p1 = permute_yields(ds, seed=3)
    p2 = permute_yields(ds, seed=3)
    assert np.array_equal(p1.y, p2.y)                     # reproducible
    assert np.array_equal(np.sort(p1.y), np.sort(ds.y))   # same multiset of yields
    assert not np.array_equal(p1.y, ds.y)                 # actually shuffled
    assert np.array_equal(p1.y_true, ds.y)                # originals kept for the audit
    assert p1.best_value == ds.best_value                 # pool optimum unchanged
    assert "permuted" in p1.title and ds.y_true is None


def test_agentic_run_writes_an_auditable_decision_log():
    ds = permute_yields(objective.sample_pool(n=100, seed=0), seed=1)
    agent = HeuristicAgent()
    run_single(ds, AgenticBO(agent), seed=0, budget=6, n_init=3, batch_size=2)
    log = agent.decision_log
    assert len(log) == 3                                  # ceil(6 / 2) rounds
    rec = log[0]
    assert rec["policy"] == "agentic_bo" and rec["round"] == 1
    assert rec["strategy"] in ("explore", "exploit") and rec["rationale"]
    assert len(rec["picked_pos"]) == 2 == rec["n_select"]
    for entry in rec["shortlist"]:
        assert entry["y"] == float(ds.y[entry["id"]])
        assert entry["y_true"] == float(ds.y_true[entry["id"]])  # permuted runs keep originals
        assert {"mean", "std", "ei"} <= set(entry)       # surrogate stats logged
    # log survives a JSON round-trip (it is written as jsonl by run.py)
    assert json.loads(json.dumps(log)) == log


def test_audit_script_reports_on_a_decision_log(tmp_path):
    ds = permute_yields(objective.sample_pool(n=100, seed=0), seed=1)
    agent = HeuristicAgent()
    run_single(ds, AgenticBO(agent), seed=0, budget=6, n_init=3, batch_size=2)
    log_path = tmp_path / "decisions.jsonl"
    log_path.write_text("\n".join(json.dumps(r) for r in agent.decision_log))
    out = subprocess.run(
        [sys.executable, os.path.join(REPO, "scripts", "audit_decisions.py"),
         str(log_path), "--show", "1"],
        capture_output=True, text=True, check=True).stdout
    assert "Strategy vs outcome" in out
    assert "Surrogate agreement" in out
    assert "Leakage indicator" in out                     # y_true present -> leakage section
    assert "Sample rationales" in out


@needs_buchwald
def test_buchwald_anonymize_withholds_reagent_identities():
    ds = reactions.load_buchwald_hartwig(anonymize=True)
    assert "anonymised" in ds.title and "withheld" in ds.legend
    for name in ("XPhos", "P2Et", "MTBD"):                # no names...
        assert name not in ds.legend
    assert "=C" not in ds.legend                          # ...and no SMILES either
    # candidates still referenced by the same short ids, so the loop is unchanged
    assert ds.descriptions == reactions.load_buchwald_hartwig().descriptions


@needs_arylation
def test_arylation_anonymize_uses_opaque_ids():
    ds = reactions.load_direct_arylation(anonymize=True)
    named = reactions.load_direct_arylation()
    assert "withheld" in ds.legend and "anonymised" in ds.title
    assert ds.descriptions[0].startswith("ligand=L")      # opaque ids replace names
    assert "temp=" in ds.descriptions[0]                  # numeric settings are kept
    assert ds.descriptions != named.descriptions
    assert np.array_equal(ds.X, named.X) and np.array_equal(ds.y, named.y)


def test_zero_shot_probe_helpers_are_deterministic():
    from leakage_probe import build_probe_prompt, sample_shortlist, score_pick

    ds = objective.sample_pool(n=80, seed=0)
    ids = sample_shortlist(ds.n, 5, [0, 7])
    assert ids == sample_shortlist(ds.n, 5, [0, 7]) and len(set(ids)) == 5
    prompt = build_probe_prompt(ds, ids)
    assert "no measurements exist yet" in prompt and "[4]" in prompt
    assert ds.descriptions[ids[0]] in prompt
    best = int(np.argmax(ds.y))
    assert score_pick(ds, best)["pool_pctile"] == 100.0


def test_default_prompt_format_is_stable():
    """The decision cache is keyed by the exact prompt — accidental prompt changes
    silently turn free cache resumes into new API calls. Pin the format."""
    ctx = DecisionContext(
        iteration=2, budget=5, remaining=4, objective_label="yield (%)",
        legend="Ligands:\n  L1 = XPhos", hist_desc=["ligand=L1"],
        y_eval=np.array([12.5]), X_eval=np.zeros((1, 2)),
        cand_desc=["ligand=L2"], cand_ids=[7], X_cand=np.zeros((1, 2)),
        mean=np.array([1.5]), std=np.array([0.25]), ei=np.array([0.125]),
        use_surrogate=True, n_select=1,
    )
    assert _build_prompt(ctx) == (
        "Experiment 2 of 5. Evaluations remaining after this one: 3.\n"
        "Objective to maximise: yield (%).\n"
        "\n"
        "Ligands:\n  L1 = XPhos\n"
        "\n"
        "Candidates measured so far (best first):\n"
        "  ligand=L1  ->  12.50\n"
        "\n"
        "Current best measured value: 12.50.\n"
        "\n"
        "Candidate shortlist with surrogate-model predictions "
        "(pred = posterior mean +/- std, EI = expected improvement):\n"
        "  [0] ligand=L2  ->  pred 1.50 +/- 0.25, EI 0.125\n"
        "\n"
        "Choose exactly one candidate by its [index], returned as a one-element 'picks' "
        "list. Prefer the pick that makes the most progress toward the global optimum "
        "given the remaining budget."
    )
