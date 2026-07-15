"""The methods compared on the linker-design task — one set, used at every realism
setting (``run_rl_design.py`` runs them with or without noise + a 3-body reward).

* ``train_policy_curve`` — train a context-``c`` REINFORCE policy, recording its exact
  value against episodes seen (open loop = context 0, closed loop = 1 or 2).
* ``model_based_planner`` — learn the reward model (Bayesian linear regression on the
  additive count features) from rollouts, then plan the optimal context-``c`` policy by
  DP. With a pairwise model + 1-block plan on the idealised task it is the sample-
  efficient "Bayesian planner"; with the wrong order/context on the realistic task it
  is the "naive planner"; with a 3-body model + 2-block plan it is "reality-aware".
* ``bo_on_plans`` — Bayesian optimisation over the pool of all fixed plans (open loop),
  with either a generic GP over one-hots or a structure-aware Bayesian-linear surrogate.
"""
from __future__ import annotations

import itertools

import numpy as np

from .design import chain_features
from .reinforce import FixedPlan, Policy
from .surrogate import BayesianLinear, GaussianSurrogate, expected_improvement


def train_policy_curve(task, context, rng, batch, eval_every, max_updates, lr, entropy,
                       target_value):
    """Train a context-``c`` policy; record exact value vs episodes; early-stop."""
    policy = Policy(task, context=context, lr=lr, entropy_coef=entropy)
    samples, values = [], []
    updates = 0
    while updates < max_updates:
        policy.train(eval_every, batch, rng)
        updates += eval_every
        samples.append(updates * batch)
        values.append(task.policy_value(policy))
        if values[-1] >= target_value:
            break
    return np.array(samples), np.array(values)


def model_based_planner(task, rng, n_episodes, order, context, warmup=12, n_points=160,
                        alpha=1e-2, beta=1.0):
    """Learn the reward model from noisy rollouts, then plan the optimal ``context``-block
    policy under it. Each episode adds one (chain, reward) observation; the Bayesian
    linear-regression posterior over the order-``order`` count features is maintained
    *incrementally* (accumulating the normal equations), so the planner can run to
    RL-scale episode counts cheaply. The reward is re-fit and re-planned at ~``n_points``
    log-spaced episode counts (fine early, sparse on the flat tail — suits a log-x plot).
    The transition model (coupling failure rate) is known; only the reward is learned — a
    misspecified model (too low an order) stays capped however many episodes it sees, a
    well-specified one converges to the optimum."""
    b, length = task.n_blocks, task.length
    d = b + b * b + (b ** 3 if order >= 3 else 0)
    xtx, xty, eye = np.zeros((d, d)), np.zeros(d), alpha * np.eye(d)
    checkpoints = set(np.unique(np.geomspace(max(warmup, 2), n_episodes, n_points)
                                .round().astype(int)))
    samples, values = [], []
    for ep in range(n_episodes):
        plan = rng.integers(0, b, size=length)
        realised = [int(rng.choice(b, p=task.transition(int(a)))) for a in plan]
        reward = task.reward(realised)
        if task.obs_noise:
            reward += float(rng.normal(0.0, task.obs_noise))
        phi = chain_features([realised], b, order=order)[0]
        xtx += np.outer(phi, phi)
        xty += phi * reward
        if (ep + 1) in checkpoints or ep + 1 == n_episodes:
            w = beta * np.linalg.solve(eye + beta * xtx, xty)   # BLR posterior mean
            unary, adj = w[:b], w[b:b + b * b].reshape(b, b)
            triplet = w[b + b * b:].reshape(b, b, b) if order >= 3 else None
            planner = task.optimal_policy(context, unary=unary, adj=adj, triplet=triplet)
            samples.append(ep + 1)
            values.append(task.policy_value(planner))
    return np.array(samples), np.array(values)


def bo_on_plans(task, rng, n_init, n_iter, replicates, structured):
    """BO over the pool of all fixed plans (open loop). ``structured`` swaps the raw
    one-hot GP for a Bayesian-linear surrogate on additive count features. Each
    evaluation averages ``replicates`` rollouts (a single one is a noisy estimate)."""
    b, length = task.n_blocks, task.length
    plans = np.array(list(itertools.product(range(b), repeat=length)))
    if structured:
        feats = chain_features(plans, b)
    else:
        feats = np.zeros((len(plans), length * b))
        feats[np.arange(len(plans))[:, None], np.arange(length)[None, :] * b + plans] = 1.0

    def observe(idx):
        return float(np.mean([task.rollout(FixedPlan(plans[idx]), rng)[2]
                              for _ in range(replicates)]))

    evaluated = list(rng.choice(len(plans), size=n_init, replace=False))
    y = [observe(i) for i in evaluated]
    samples, values = [], []
    for _ in range(n_iter):
        surr = (BayesianLinear() if structured else GaussianSurrogate(seed=0))
        surr.fit(feats[evaluated], np.array(y))
        mean, std = surr.predict(feats)
        ei = expected_improvement(mean, std, best=max(y))
        ei[evaluated] = -np.inf
        pick = int(np.argmax(ei))
        evaluated.append(pick)
        y.append(observe(pick))
        # The structured model generalises exactly, so its pool-wide argmax is
        # trustworthy; the raw-one-hot GP is not, so recommend among evaluated only.
        deployed = int(np.argmax(mean)) if structured \
            else evaluated[int(np.argmax(mean[evaluated]))]
        samples.append(len(evaluated) * replicates)
        values.append(task.plan_expected_reward(plans[deployed]))
    return np.array(samples), np.array(values)
