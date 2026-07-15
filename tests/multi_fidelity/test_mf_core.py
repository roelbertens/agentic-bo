"""Offline tests for the multi-fidelity core: oracles, ledger, curves, baselines."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from datasets import synthetic
from multi_fidelity.baselines import MFBO, ClassicBOHF, RandomHF, TwoStage
from multi_fidelity.campaign import CampaignState, run_campaign, run_method
from multi_fidelity.oracles import (
    BudgetExceeded,
    MultiFidelityProblem,
    best_hf_curve,
    hf_share_curve,
)


def make_problem(rho=0.8, n=200, cost_lf=1.0, cost_hf=10.0, seed=0):
    return MultiFidelityProblem(synthetic.sample_pool(n=n, seed=0), rho=rho,
                                cost_lf=cost_lf, cost_hf=cost_hf, seed=seed)


def test_lf_correlation_tracks_rho():
    big = synthetic.sample_pool(n=2000, seed=0)
    for rho in (0.9, 0.5, 0.1):
        prob = MultiFidelityProblem(big, rho=rho)
        assert abs(prob.empirical_rho() - rho) < 0.08
    # rho = 1 is the truth itself; the corruption is seeded and reproducible
    assert np.allclose(MultiFidelityProblem(big, rho=1.0).y_lf, big.y)
    assert np.allclose(MultiFidelityProblem(big, rho=0.4, seed=3).y_lf,
                       MultiFidelityProblem(big, rho=0.4, seed=3).y_lf)


def test_ledger_enforces_the_budget():
    prob = make_problem()
    state = CampaignState(prob, budget=25.0)
    state.measure([0, 1])                      # 20
    state.simulate([2, 3, 4])                  # 23
    with pytest.raises(BudgetExceeded):
        state.measure([5])                     # 33 > 25
    assert state.ledger.spent == 23.0          # the failed call charged nothing
    costs = [o.cost_after for o in state.ledger.log]
    assert costs == sorted(costs) and costs[-1] == 23.0


def test_best_curve_counts_only_measurements():
    prob = make_problem()
    state = CampaignState(prob, budget=50.0)
    top = int(np.argmax(prob.y_lf))
    state.simulate([top])                      # a great simulated value is not a result
    lo, hi = np.argsort(prob.dataset.y)[[10, -1]]
    state.measure([int(lo), int(hi)])
    curve = best_hf_curve(state.ledger.log, 50.0)
    assert np.isnan(curve[0])                                    # nothing measured yet
    assert curve[-1] == prob.dataset.y[hi]
    valid = curve[~np.isnan(curve)]
    assert np.all(np.diff(valid) >= 0)                           # best-so-far is monotone
    share = hf_share_curve(state.ledger.log, prob, 50.0)
    assert share[-1] == pytest.approx(20.0 / 21.0)               # 2 HF of 21 total spend


@pytest.mark.parametrize("policy_cls", [RandomHF, ClassicBOHF, TwoStage, MFBO])
def test_policies_spend_within_budget_until_dry(policy_cls):
    prob = make_problem(n=150)
    state = run_campaign(prob, policy_cls(), seed=0, budget=100.0, n_init=3)
    assert state.ledger.spent <= 100.0 + 1e-9
    # dry = another measurement is unaffordable (or the pool ran out)
    assert not state.can_measure() or not len(state.unmeasured())
    assert len(state.hf_obs) >= 3


def test_classic_bo_beats_random():
    prob = make_problem(n=400)
    seeds = range(4)
    rand = run_method(prob, RandomHF, "random", seeds, 120.0, 3, verbose=False)
    bo = run_method(prob, ClassicBOHF, "bo", seeds, 120.0, 3, verbose=False)
    assert bo.curves[:, -1].mean() > rand.curves[:, -1].mean()


def test_mfbo_uses_the_simulator_when_it_is_cheap():
    prob = make_problem(rho=0.9, n=150)
    state = run_campaign(prob, MFBO(), seed=1, budget=100.0, n_init=3)
    assert len(state.lf_obs) > 0
