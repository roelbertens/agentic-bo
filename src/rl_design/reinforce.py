"""REINFORCE for the sequential linker-design task (``design.py``).

A from-scratch policy gradient — no autodiff, because the policy is a tabular softmax
whose gradient is closed form. One ``Policy`` covers every regime through its
**context** ``c`` (how many previous realised blocks it conditions on):

* ``c = 0`` — open loop: conditions on the step only, a plan that ignores what actually
  happened. Bounded by the best fixed plan.
* ``c = 1`` — closed loop on the last block: adapts to a mis-coupling.
* ``c = 2`` — closed loop on the last two blocks: enough for a 3-body reward.

Training is vanilla REINFORCE: roll out a batch of episodes, use the terminal return
with a batch-mean baseline and standardised advantages, and ascend
``advantage * grad log pi`` with a small entropy bonus for exploration.
"""
from __future__ import annotations

import numpy as np

from .design import DesignTask, _encode


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


class Policy:
    """Softmax REINFORCE policy conditioned on the last ``context`` realised blocks."""

    def __init__(self, task: DesignTask, context: int = 1, lr: float = 0.5,
                 entropy_coef: float = 0.02):
        self.task = task
        self.context = context
        self.lr = lr
        self.entropy_coef = entropy_coef
        b = task.n_blocks
        self.base = b + 1
        self.theta = np.zeros((task.length, self.base ** context, b))

    def _row(self, t, hist):
        return self.theta[t, _encode(hist, self.base)]

    def action(self, t, hist, rng):
        return int(rng.choice(self.task.n_blocks, p=_softmax(self._row(t, hist))))

    def greedy(self, t, hist):
        return int(np.argmax(self._row(t, hist)))

    def train(self, updates, batch, rng):
        for _ in range(updates):
            grad = np.zeros_like(self.theta)
            episodes, returns = [], np.empty(batch)
            for i in range(batch):
                histories, actions, reward = self.task.rollout(self, rng)
                episodes.append((histories, actions))
                returns[i] = reward
            adv = (returns - returns.mean()) / (returns.std() + 1e-8)
            for (histories, actions), a_i in zip(episodes, adv, strict=True):
                for t, (hist, act) in enumerate(zip(histories, actions, strict=True)):
                    idx = _encode(hist, self.base)
                    p = _softmax(self.theta[t, idx])
                    onehot = np.zeros_like(p)
                    onehot[act] = 1.0
                    entropy = -float((p * np.log(p + 1e-12)).sum())
                    ent_grad = -p * (np.log(p + 1e-12) + entropy)   # d H / d logits
                    grad[t, idx] += a_i * (onehot - p) + self.entropy_coef * ent_grad
            self.theta += self.lr * grad / batch


class FixedPlan:
    """An open-loop policy: install a fixed sequence of blocks, ignoring the state.

    This is what a Bayesian-optimisation baseline searches over — one fixed plan is one
    candidate in the pool of all ``n_blocks ** length`` plans.
    """

    context = 0

    def __init__(self, plan):
        self.plan = [int(a) for a in plan]

    def action(self, t, hist, rng):
        return self.plan[t]

    def greedy(self, t, hist):
        return self.plan[t]
