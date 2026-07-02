"""Fast smoke tests — run the whole pipeline with the heuristic agent (no API)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agentic_bo import objective, reactions
from agentic_bo.agent import HeuristicAgent
from agentic_bo.experiment import run_method, run_single
from agentic_bo.policies import AgenticBO, ClassicBO, RandomPolicy


def test_dataset_shapes():
    ds = objective.sample_pool(n=100, seed=0)
    assert ds.X.shape == (100, 5)
    assert ds.y.shape == (100,)
    # sampling is reproducible
    assert np.allclose(ds.y, objective.sample_pool(n=100, seed=0).y)


def test_curves_are_monotone_and_bounded():
    ds = objective.sample_pool(n=200, seed=0)
    for policy in [RandomPolicy(), ClassicBO(), AgenticBO(HeuristicAgent())]:
        curve = run_single(ds, policy, seed=1, budget=8, n_init=4)
        assert len(curve) == 12
        assert np.all(np.diff(curve) >= 0)          # best-so-far never decreases
        assert curve[-1] <= ds.best_value + 1e-9    # cannot beat the pool optimum


def test_bo_beats_random_on_average():
    ds = objective.sample_pool(n=400, seed=0)
    seeds = range(6)
    rand = run_method(ds, lambda: RandomPolicy(), "random", seeds, 15, 5, verbose=False)
    bo = run_method(ds, lambda: ClassicBO(), "classic_bo", seeds, 15, 5, verbose=False)
    assert bo.curves[:, -1].mean() > rand.curves[:, -1].mean()


def test_surrogate_helps_the_agent():
    ds = objective.sample_pool(n=400, seed=0)
    seeds = range(6)
    with_surr = run_method(ds, lambda: AgenticBO(HeuristicAgent(), use_surrogate=True),
                           "agentic_bo", seeds, 15, 5, verbose=False)
    without = run_method(ds, lambda: AgenticBO(HeuristicAgent(), use_surrogate=False),
                         "agentic_no_surrogate", seeds, 15, 5, verbose=False)
    assert with_surr.curves[:, -1].mean() > without.curves[:, -1].mean()


@pytest.mark.skipif(
    not os.path.exists(os.path.join(os.path.dirname(__file__), "..", "data", "buchwald_hartwig.xlsx")),
    reason="run scripts/get_data.py to fetch the Buchwald-Hartwig dataset",
)
def test_buchwald_loader_and_run():
    ds = reactions.load_buchwald_hartwig()
    assert ds.n == 3955
    assert ds.X.shape[1] == 4 + 3 + 22 + 15   # one-hot over the four reagent categories
    assert 0.0 <= ds.y.min() and ds.y.max() <= 130.0
    assert len(ds.descriptions) == ds.n and ds.legend
    curve = run_single(ds, AgenticBO(HeuristicAgent()), seed=0, budget=8, n_init=4)
    assert np.all(np.diff(curve) >= 0) and curve[-1] <= ds.best_value + 1e-9


@pytest.mark.skipif(
    not os.path.exists(os.path.join(os.path.dirname(__file__), "..", "data",
                                    "direct_arylation", "experiment_index.csv")),
    reason="run scripts/get_data.py to fetch the direct-arylation dataset",
)
def test_direct_arylation_loader_and_run():
    ds = reactions.load_direct_arylation()
    assert ds.n == 1728
    assert ds.X.shape[1] == 12 + 4 + 4 + 2   # one-hot ligand/base/solvent + conc + temp
    assert len(ds.descriptions) == ds.n and "ligand=" in ds.descriptions[0]
    curve = run_single(ds, AgenticBO(HeuristicAgent()), seed=0, budget=8, n_init=4)
    assert np.all(np.diff(curve) >= 0) and curve[-1] <= ds.best_value + 1e-9
