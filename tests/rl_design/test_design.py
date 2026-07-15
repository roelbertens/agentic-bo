"""Tests for the unified linker-design task: env, exact DP, and the methods — offline.

Reality is a dial (``triplet_strength``, ``obs_noise``); the same code path covers the
idealised and realistic settings, so both are exercised here through those knobs.
"""
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from rl_design.design import DesignTask, chain_features
from rl_design.methods import bo_on_plans, model_based_planner, train_policy_curve
from rl_design.reinforce import FixedPlan, Policy
from rl_design.surrogate import BayesianLinear

# --- environment ------------------------------------------------------------------

def test_reward_matches_manual_and_triplet_and_noise():
    task = DesignTask.default()
    seq = [3, 2, 3]  # NH2, COOH, NH2 -> two complementary donor/acceptor bonds
    expected = (task.unary[3] + task.unary[2] + task.unary[3]
                + task.adj[3, 2] + task.adj[2, 3])
    assert np.isclose(task.reward(seq), expected)
    # a 3-body term changes the reward; the field is the right shape
    witht = DesignTask.default(triplet_strength=1.0)
    assert witht.triplet.shape == (6, 6, 6)
    assert not np.isclose(task.reward([0, 2, 3, 1, 4]), witht.reward([0, 2, 3, 1, 4]))
    # obs_noise perturbs the observed rollout reward but not reward()
    noisy = DesignTask.default(obs_noise=1.0)
    _, _, r = noisy.rollout(FixedPlan([0] * noisy.length), np.random.default_rng(0))
    assert not np.isclose(r, noisy.reward([0] * noisy.length))


def test_chain_features_counts_blocks_pairs_triples():
    f2 = chain_features([0, 0, 1], n_blocks=3)
    assert f2.shape == (1, 3 + 9)
    assert list(f2[0, :3]) == [2, 1, 0]
    f3 = chain_features([0, 0, 1], n_blocks=3, order=3)
    assert f3.shape == (1, 3 + 9 + 27)
    triples = f3[0, 12:].reshape(3, 3, 3)
    assert triples[0, 0, 1] == 1 and triples.sum() == 1  # one triplet (0,0,1)


def test_transition_is_a_distribution():
    task = DesignTask.default(success_prob=0.6)
    d = task.transition(2)
    assert np.isclose(d.sum(), 1.0)
    assert np.isclose(d[2], 0.6 + 0.4 / task.n_blocks)


# --- exact references (DP) --------------------------------------------------------

def test_optimal_policy_achieves_the_optimum():
    task = DesignTask.default()
    policy = task.optimal_policy(context=1)
    assert np.isclose(task.policy_value(policy), task.optimal_value(1))
    assert task.optimal_value(1) > task.random_value() + 1.0


def test_open_loop_optimum_matches_brute_force_and_is_below_closed():
    task = DesignTask.default(length=3)
    brute = max(task.plan_expected_reward(p)
                for p in itertools.product(range(task.n_blocks), repeat=task.length))
    assert np.isclose(task.open_loop_optimum(), brute)
    full = DesignTask.default()
    assert full.open_loop_optimum() < full.optimal_value(1) - 0.5


def test_context2_dp_matches_monte_carlo_with_triplet():
    task = DesignTask.default(length=4, triplet_strength=0.8)
    policy = task.optimal_policy(2, triplet=task.triplet)
    exact = task.policy_value(policy)
    rng = np.random.default_rng(0)
    b, total, n = task.n_blocks, 0.0, 40000
    for _ in range(n):
        hist, realised = (b, b), []
        for t in range(task.length):
            r = int(rng.choice(b, p=task.transition(policy.greedy(t, hist))))
            realised.append(r)
            hist = (hist[-1], r)
        total += task.reward(realised)
    assert abs(exact - total / n) < 0.1


def test_richer_context_helps_only_with_the_3body_reward():
    plain = DesignTask.default()
    assert np.isclose(plain.optimal_value(2), plain.optimal_value(1), atol=1e-6)
    complex_ = DesignTask.default(triplet_strength=1.0)
    assert complex_.optimal_value(2) > complex_.optimal_value(1) + 0.1


def test_open_loop_references_include_the_3body_term():
    task = DesignTask.default(length=4, triplet_strength=1.0)
    # a plan's exact expected reward matches Monte Carlo on the triplet task
    plan = np.array([3, 2, 0, 1])
    rng = np.random.default_rng(0)
    n = 200_000
    fails = rng.random((n, task.length)) >= task.success_prob
    r = np.where(fails, rng.integers(0, task.n_blocks, size=(n, task.length)),
                 plan[None, :])
    mc = (task.unary[r].sum(axis=1) + task.adj[r[:, :-1], r[:, 1:]].sum(axis=1)
          + task.triplet[r[:, :-2], r[:, 1:-1], r[:, 2:]].sum(axis=1)).mean()
    assert abs(task.plan_expected_reward(plan) - mc) < 0.05
    # and the open-loop optimum matches brute force over every plan
    brute = max(task.plan_expected_reward(p)
                for p in itertools.product(range(task.n_blocks), repeat=task.length))
    assert np.isclose(task.open_loop_optimum(), brute)


# --- surrogate --------------------------------------------------------------------

def test_bayesian_linear_recovers_a_linear_function():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 5))
    w = np.array([1.0, -2.0, 0.5, 0.0, 3.0])
    y = x @ w + rng.normal(scale=0.01, size=200)
    blr = BayesianLinear(alpha=1e-6, beta=1e4).fit(x, y)
    assert np.allclose(blr.weights(), w, atol=0.05)


# --- learners ---------------------------------------------------------------------

def test_policy_learns_above_random_at_each_context():
    task = DesignTask.default()
    for context in (0, 1):
        policy = Policy(task, context=context)
        policy.train(120, 64, np.random.default_rng(0))
        assert task.policy_value(policy) > task.random_value() + 1.0


def test_context2_policy_learns_and_handles_the_start_token():
    task = DesignTask.default()
    policy = Policy(task, context=2)
    start = (task.n_blocks, task.n_blocks)          # both history slots = start token
    assert 0 <= policy.greedy(0, start) < task.n_blocks
    policy.train(150, 64, np.random.default_rng(0))
    assert task.policy_value(policy) > task.random_value() + 1.0


def test_training_is_deterministic_for_a_seed():
    task = DesignTask.default()
    a = train_policy_curve(task, 1, np.random.default_rng(3), 32, 10, 40, 0.5, 0.02, 1e9)
    b = train_policy_curve(task, 1, np.random.default_rng(3), 32, 10, 40, 0.5, 0.02, 1e9)
    assert np.allclose(a[1], b[1])


def test_train_policy_curve_stops_at_its_target():
    task = DesignTask.default()
    args = (32, 10, 60, 0.5, 0.02)
    stopped = train_policy_curve(task, 1, np.random.default_rng(3), *args,
                                 task.random_value())   # trivially reachable target
    capped = train_policy_curve(task, 1, np.random.default_rng(3), *args, 1e9)
    assert len(stopped[0]) < len(capped[0])
    assert stopped[1][-1] >= task.random_value()


# --- methods ----------------------------------------------------------------------

def test_bo_on_plans_reaches_the_fixed_plan_ceiling():
    task = DesignTask.default()
    _, values = bo_on_plans(task, np.random.default_rng(1), 6, 44, replicates=20,
                            structured=True)
    assert values[-1] > 0.9 * task.open_loop_optimum()


def test_bo_deployed_plan_never_beats_the_open_loop_optimum():
    task = DesignTask.default(length=3)             # 216 plans: exercises the GP path fast
    for structured in (True, False):
        _, values = bo_on_plans(task, np.random.default_rng(0), 5, 8, replicates=5,
                                structured=structured)
        assert np.all(values <= task.open_loop_optimum() + 1e-9)


def test_model_based_planner_reaches_optimum_on_the_idealised_task():
    task = DesignTask.default()
    _, values = model_based_planner(task, np.random.default_rng(1), 120, order=2,
                                    context=1)
    assert values[-1] > 0.95 * task.optimal_value(1)


def test_planner_checkpoints_cover_the_full_budget():
    task = DesignTask.default()
    samples, values = model_based_planner(task, np.random.default_rng(0), 500, order=2,
                                          context=1, warmup=10, n_points=30)
    assert samples[0] >= 2 and samples[-1] == 500
    assert np.all(np.diff(samples) > 0)             # strictly increasing episode counts
    assert len(samples) == len(values) and np.all(np.isfinite(values))


def test_sim_to_real_gap_and_recovery():
    task = DesignTask.default(triplet_strength=1.0, obs_noise=0.5)
    # a pairwise model with a 1-block state is capped below the true 3-body optimum
    _, naive = model_based_planner(task, np.random.default_rng(1), 120, order=2,
                                   context=1, warmup=60)
    assert naive[-1] < task.optimal_value(2) - 0.5
    # a 3-body model with a 2-block state recovers past the 1-block ceiling
    _, aware = model_based_planner(task, np.random.default_rng(1), 500, order=3,
                                   context=2, warmup=80)
    assert aware[-1] > task.optimal_value(1)
    assert aware[-1] > 0.95 * task.optimal_value(2)


def test_misspecified_model_is_capped_regardless_of_state():
    # more state does not fix a wrong model: a pairwise model given the full 2-block
    # state still cannot represent the 3-body reward and stays capped below the optimum
    task = DesignTask.default(triplet_strength=1.0)
    _, capped = model_based_planner(task, np.random.default_rng(1), 400, order=2,
                                    context=2, warmup=80)
    assert capped[-1] > task.random_value() + 1.0   # it does learn the pairwise part
    assert capped[-1] < task.optimal_value(2) - 0.5
    # and the extra state is exactly inert: without a triplet term the DP's value
    # recursion never reads the second-back block, so the 2-block plan collapses to
    # the 1-block plan — same policy value, not merely a similar one
    v1 = task.policy_value(task.optimal_policy(1, triplet=None))
    v2 = task.policy_value(task.optimal_policy(2, triplet=None))
    assert np.isclose(v1, v2, atol=1e-12)
