"""A sequential molecular-design task: build a linker one block at a time.

This is a genuine episodic Markov decision process, not a selection from a fixed
pool. An episode assembles a chain of ``length`` building blocks left to right. At
each step the agent picks a block to install; the installed block is the one it
chose only when the coupling *succeeds* (probability ``success_prob``), and otherwise
a uniformly random block ends up in that position — a mis-coupling. The reward, paid
at the end, scores the *realised* chain.

Reality is a dial, not a separate task. The reward always has per-block and adjacent
(``adj``) terms; two optional knobs make it more realistic:

* ``triplet`` — a 3-body reward term a pairwise model cannot represent (the "complex
  reality"); off by default.
* ``obs_noise`` — Gaussian measurement noise on the verifier; off by default.

Everything is parameterised by a policy's **context** ``c`` — how many previous
realised blocks it conditions on: ``c = 0`` open loop (a fixed plan), ``1`` = the last
block, ``2`` = the last two (enough for a 3-body reward). Because the vocabulary and
length are small, the exact expected value of any deterministic policy — and the
optimum at each context — is computed by backward DP over the last two blocks, giving
an honest "% of optimum" scale that needs no training.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# A toy vocabulary with chemistry-flavoured names. Groups drive the adjacency term.
BLOCKS = ["Ph", "Py", "COOH", "NH2", "OH", "spacer"]
_DONORS = {"NH2", "OH"}
_ACCEPTORS = {"COOH", "Py"}
_AROMATIC = {"Ph"}
_SPACER = {"spacer"}

# Per-block contribution (a mild preference, dominated by adjacency below).
_UNARY = {"COOH": 0.5, "Py": 0.3, "NH2": 0.2, "OH": 0.2, "Ph": 0.1, "spacer": -0.2}


def chain_features(sequences, n_blocks: int, order: int = 2) -> np.ndarray:
    """Structured features that make the reward *linear*: per-block counts,
    adjacent-pair counts, and (``order=3``) adjacent-triplet counts of a sequence.

    A chain's reward is a sum of per-block and neighbour terms, so it equals
    ``counts . unary + pair_counts . adjacency`` — linear in these features. A model
    on them (see ``surrogate.BayesianLinear``) is therefore an exact surrogate, unlike
    a generic GP over raw one-hots. ``order=3`` also makes a 3-body reward linear.
    Accepts one sequence or a batch; returns ``(n, n_blocks + n_blocks**2)`` (order 2)
    or ``+ n_blocks**3`` (order 3).
    """
    seqs = np.atleast_2d(np.asarray(sequences, dtype=int))
    n, length = seqs.shape
    counts = np.zeros((n, n_blocks))
    for b in range(n_blocks):
        counts[:, b] = (seqs == b).sum(axis=1)
    pairs = np.zeros((n, n_blocks * n_blocks))
    for t in range(length - 1):
        np.add.at(pairs, (np.arange(n), seqs[:, t] * n_blocks + seqs[:, t + 1]), 1.0)
    blocks = [counts, pairs]
    if order >= 3:
        triples = np.zeros((n, n_blocks ** 3))
        for t in range(length - 2):
            idx = (seqs[:, t] * n_blocks + seqs[:, t + 1]) * n_blocks + seqs[:, t + 2]
            np.add.at(triples, (np.arange(n), idx), 1.0)
        blocks.append(triples)
    return np.hstack(blocks)


def build_triplet(n_blocks: int, strength: float, seed: int = 12345) -> np.ndarray:
    """A fixed random 3-body reward tensor ``T[a, b, c]`` scaled by ``strength`` — the
    'complex reality' a pairwise model and a last-one-block state cannot represent."""
    rng = np.random.default_rng(seed)
    return strength * rng.normal(size=(n_blocks, n_blocks, n_blocks))


def _adjacency(a: str, b: str) -> float:
    """Reward for placing block ``b`` immediately after block ``a`` (symmetric)."""
    if a in _SPACER or b in _SPACER:
        return 0.0                                   # a spacer is inert to its neighbours
    if (a in _DONORS and b in _ACCEPTORS) or (a in _ACCEPTORS and b in _DONORS):
        return 2.0                                   # complementary donor/acceptor pair
    if (a in _DONORS and b in _DONORS) or (a in _ACCEPTORS and b in _ACCEPTORS):
        return -1.0                                  # like-next-to-like repels
    if a in _AROMATIC and b in _AROMATIC:
        return 1.0                                   # pi-stacking
    return 0.3                                        # aromatic next to a polar group


def _build_matrices() -> tuple[np.ndarray, np.ndarray]:
    unary = np.array([_UNARY[b] for b in BLOCKS], dtype=float)
    adj = np.array([[_adjacency(a, b) for b in BLOCKS] for a in BLOCKS], dtype=float)
    return unary, adj


def _encode(hist, base: int) -> int:
    idx = 0
    for h in hist:
        idx = idx * base + int(h)
    return idx


class TablePolicy:
    """A deterministic policy given by a table over (step, last-``context`` blocks).

    ``optimal_policy`` returns one of these; a model-based planner turns its learned
    reward model into one by DP.
    """

    def __init__(self, table: np.ndarray, context: int, base: int):
        self.table = table            # (length, base**context)
        self.context = context
        self.base = base

    def greedy(self, t: int, hist) -> int:
        return int(self.table[t, _encode(hist, self.base)])

    def action(self, t: int, hist, rng) -> int:
        return self.greedy(t, hist)


@dataclass(frozen=True)
class DesignTask:
    """The linker-design MDP. ``unary``/``adj``/``triplet`` define the verifier's
    reward; ``obs_noise`` adds measurement noise to observed rollouts."""

    length: int = 6
    success_prob: float = 0.6
    unary: np.ndarray = None       # (B,) per-block reward
    adj: np.ndarray = None         # (B, B) neighbour reward
    triplet: np.ndarray = None     # optional (B, B, B) 3-body reward
    obs_noise: float = 0.0         # optional Gaussian measurement noise

    @staticmethod
    def default(length: int = 6, success_prob: float = 0.6,
                triplet_strength: float = 0.0, obs_noise: float = 0.0) -> DesignTask:
        unary, adj = _build_matrices()
        triplet = (build_triplet(len(unary), triplet_strength)
                   if triplet_strength > 0 else None)
        return DesignTask(length=length, success_prob=success_prob, unary=unary, adj=adj,
                          triplet=triplet, obs_noise=obs_noise)

    @property
    def n_blocks(self) -> int:
        return len(self.unary)

    def transition(self, action: int) -> np.ndarray:
        """Distribution over the realised block given the chosen ``action``: success
        installs it, a failed coupling installs a uniformly random block."""
        b = self.n_blocks
        dist = np.full(b, (1.0 - self.success_prob) / b)
        dist[action] += self.success_prob
        return dist

    def transition_matrix(self) -> np.ndarray:
        """Row ``a`` = distribution over the realised block for chosen action ``a``."""
        b = self.n_blocks
        m = np.full((b, b), (1.0 - self.success_prob) / b)
        m[np.arange(b), np.arange(b)] += self.success_prob
        return m

    def reward(self, realised) -> float:
        """Verifier score of a fully realised chain of block indices (noiseless)."""
        r = np.asarray(realised, dtype=int)
        total = float(self.unary[r].sum()) + float(self.adj[r[:-1], r[1:]].sum())
        if self.triplet is not None and len(r) >= 3:
            total += float(self.triplet[r[:-2], r[1:-1], r[2:]].sum())
        return total

    def rollout(self, policy, rng: np.random.Generator):
        """One episode under ``policy``; returns (histories, actions, noisy reward).

        ``histories[t]`` is the tuple of the last ``policy.context`` realised blocks the
        policy saw at step t (padded with the start token = ``n_blocks``) — what a
        REINFORCE update needs for credit assignment.
        """
        b, c = self.n_blocks, policy.context
        hist = (b,) * c
        realised, histories, actions = [], [], []
        for t in range(self.length):
            a = policy.action(t, hist, rng)
            r = int(rng.choice(b, p=self.transition(a)))
            histories.append(hist)
            actions.append(a)
            realised.append(r)
            hist = (hist[1:] + (r,)) if c > 0 else ()
        reward = self.reward(realised)
        if self.obs_noise:
            reward += float(rng.normal(0.0, self.obs_noise))
        return histories, actions, reward

    # --- exact references, by backward DP over the last two blocks -----------------
    #
    # Tracking the last two realised blocks is enough to evaluate a 3-body reward and
    # any policy of context <= 2. ``prev = n_blocks`` is the start token.

    def policy_value(self, policy) -> float:
        """Exact expected return of a deterministic ``policy`` under the true reward."""
        b, tm, tri = self.n_blocks, self.transition_matrix(), self.triplet
        b1 = b + 1
        v = np.zeros((self.length + 1, b1, b1))
        for t in range(self.length - 1, -1, -1):
            for p2 in range(b1):
                for p1 in range(b1):
                    hist = () if policy.context == 0 else (
                        (p1,) if policy.context == 1 else (p2, p1))
                    a = policy.greedy(t, hist)
                    neigh = np.zeros(b) if p1 == b else self.adj[p1]
                    trip = (np.zeros(b) if (tri is None or p2 == b or p1 == b)
                            else tri[p2, p1])
                    immediate = self.unary + neigh + trip + v[t + 1, p1, :b]
                    v[t, p2, p1] = float(tm[a] @ immediate)
        return float(v[0, b, b])

    def random_value(self) -> float:
        """Exact expected return of the uniform-random policy (true reward)."""
        b, tri, avg = self.n_blocks, self.triplet, self.transition_matrix().mean(axis=0)
        b1 = b + 1
        v = np.zeros((self.length + 1, b1, b1))
        for t in range(self.length - 1, -1, -1):
            for p2 in range(b1):
                for p1 in range(b1):
                    neigh = np.zeros(b) if p1 == b else self.adj[p1]
                    trip = (np.zeros(b) if (tri is None or p2 == b or p1 == b)
                            else tri[p2, p1])
                    v[t, p2, p1] = float(avg @ (self.unary + neigh + trip + v[t + 1, p1, :b]))
        return float(v[0, b, b])

    def optimal_policy(self, context: int = 1, unary=None, adj=None,
                       triplet=None) -> TablePolicy:
        """Optimal policy at a given ``context`` under a (possibly learned) reward model.

        ``context = 1`` plans over the last block (a pairwise planner ignores any
        triplet it was not given); ``context = 2`` plans over the last two and can use a
        triplet model.
        """
        unary = self.unary if unary is None else unary
        adj = self.adj if adj is None else adj
        b, tm, b1 = self.n_blocks, self.transition_matrix(), self.n_blocks + 1
        if context == 1:
            v = np.zeros((self.length + 1, b1))
            act = np.zeros((self.length, b1), dtype=int)
            for t in range(self.length - 1, -1, -1):
                for p1 in range(b1):
                    neigh = np.zeros(b) if p1 == b else adj[p1]
                    q = tm @ (unary + neigh + v[t + 1, :b])
                    act[t, p1] = int(np.argmax(q))
                    v[t, p1] = float(q.max())
            return TablePolicy(act, 1, b1)
        v = np.zeros((self.length + 1, b1, b1))
        act = np.zeros((self.length, b1, b1), dtype=int)
        for t in range(self.length - 1, -1, -1):
            for p2 in range(b1):
                for p1 in range(b1):
                    neigh = np.zeros(b) if p1 == b else adj[p1]
                    trip = (np.zeros(b) if (triplet is None or p2 == b or p1 == b)
                            else triplet[p2, p1])
                    q = tm @ (unary + neigh + trip + v[t + 1, p1, :b])
                    act[t, p2, p1] = int(np.argmax(q))
                    v[t, p2, p1] = float(q.max())
        return TablePolicy(act.reshape(self.length, b1 * b1), 2, b1)

    def optimal_value(self, context: int = 1) -> float:
        """Value of the best policy at ``context`` (context 2 = the true optimum)."""
        tri = self.triplet if context >= 2 else None
        return self.policy_value(self.optimal_policy(context, triplet=tri))

    # --- open-loop (fixed-plan) reference, for the Bayesian-optimisation baseline ---
    #
    # A fixed plan is a sequence of actions chosen up front. Because the realised blocks
    # are independent given the actions, a plan's *expected* reward is a chain in action
    # space (each reward term contracts against the per-action realisation distributions),
    # so the best plan and any plan's value are exact and cheap — with a 3-body reward the
    # DP state is the last two actions instead of one. These bound what an optimiser over
    # fixed plans (e.g. BO) can achieve.

    def _expected_action_terms(self) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        d = self.transition_matrix()                 # row a = distribution over realised block
        g = d @ self.unary                           # (B,) expected per-block reward of action a
        h = d @ self.adj @ d.T                        # (B, B) expected adjacency for action pair
        h3 = (np.einsum("ai,bj,ck,ijk->abc", d, d, d, self.triplet)
              if self.triplet is not None else None)  # (B, B, B) expected 3-body term
        return g, h, h3

    def plan_expected_reward(self, plan) -> float:
        """Exact expected reward of executing a fixed action sequence (open loop)."""
        g, h, h3 = self._expected_action_terms()
        a = np.asarray(plan, dtype=int)
        total = float(g[a].sum() + h[a[:-1], a[1:]].sum())
        if h3 is not None and len(a) >= 3:
            total += float(h3[a[:-2], a[1:-1], a[2:]].sum())
        return total

    def open_loop_optimum(self) -> float:
        """Best expected reward over all fixed plans (exact, via DP over actions)."""
        g, h, h3 = self._expected_action_terms()
        if h3 is None or self.length < 3:
            v = g.copy()
            for _ in range(self.length - 1):
                v = g + (h + v[None, :]).max(axis=1)  # V_t(a) = g[a] + max_a' (H + V_{t+1})
            return float(v.max())
        v = np.zeros((self.n_blocks, self.n_blocks))  # V_t(a_{t-2}, a_{t-1})
        for _ in range(self.length - 2):
            v = (g[None, None, :] + h[None, :, :] + h3 + v[None, :, :]).max(axis=2)
        return float((g[:, None] + g[None, :] + h + v).max())

    def describe(self, realised) -> str:
        return " - ".join(BLOCKS[i] for i in np.asarray(realised, dtype=int))
