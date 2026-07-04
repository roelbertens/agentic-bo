"""Optimisation policies: what to measure next given what we've measured so far.

All policies share the ``propose`` interface and operate on a fixed candidate
pool (pool-based BO, i.e. screening from a finite library).

* ``RandomPolicy``  — naive floor.
* ``ClassicBO``     — the baseline: GP surrogate + Expected Improvement
  (local-penalization batching for q > 1).
* ``AgenticBO``     — an LLM/heuristic agent chooses from a surrogate-informed
  shortlist. With ``use_surrogate=False`` it becomes the *ablation*: the same
  agent, but without the GP's mean/std/EI to reason over.
"""
from __future__ import annotations

import numpy as np

from .agent import DecisionContext
from .data import Dataset
from .surrogate import GaussianSurrogate, expected_improvement


class RandomPolicy:
    name = "random"

    def propose_batch(self, dataset, evaluated, y_obs, iteration, budget, rng, seed, q=1):
        unseen = _unseen(dataset, evaluated)
        q = min(q, len(unseen))
        return [int(i) for i in rng.choice(unseen, size=q, replace=False)]


class ClassicBO:
    name = "classic_bo"

    def propose_batch(self, dataset, evaluated, y_obs, iteration, budget, rng, seed, q=1):
        unseen = _unseen(dataset, evaluated)
        mean, std = _fit_predict(dataset, evaluated, y_obs, unseen, seed)
        ei = expected_improvement(mean, std, best=float(np.max(y_obs)))
        if q == 1:
            return [int(unseen[int(np.argmax(ei))])]
        # Batch via local penalization: after each pick, suppress the acquisition of
        # nearby candidates so the batch spreads out (a light stand-in for qLogEI, and
        # a far fairer batch baseline than picking the top-q near-duplicate EI points).
        Xc = dataset.X[unseen]
        picks = _greedy_batch_lp(Xc, ei, min(q, len(unseen)), _length_scale(Xc, rng))
        return [int(unseen[j]) for j in picks]


class AgenticBO:
    """Agent-driven policy. The agent picks q candidates from a shortlist each round."""

    def __init__(self, agent, use_surrogate: bool = True, shortlist_size: int = 8):
        self.agent = agent
        self.use_surrogate = use_surrogate
        self.shortlist_size = shortlist_size
        self.name = "agentic_bo" if use_surrogate else "agentic_no_surrogate"

    def propose_batch(self, dataset, evaluated, y_obs, iteration, budget, rng, seed, q=1):
        unseen = _unseen(dataset, evaluated)
        # Shortlist must be big enough to choose q from, with room to reason.
        k = min(max(self.shortlist_size, q + 4), len(unseen))
        mean = std = ei = None

        if self.use_surrogate:
            m, s = _fit_predict(dataset, evaluated, y_obs, unseen, seed)
            e = expected_improvement(m, s, best=float(np.max(y_obs)))
            # Shortlist = mostly top-EI candidates, plus a couple of high-uncertainty
            # picks so the agent always has an exploration option on the table.
            n_explore = min(2, k)
            picks: list[int] = list(np.argsort(-e)[: k - n_explore])
            for j in np.argsort(-s):
                if len(picks) >= k:
                    break
                if int(j) not in picks:
                    picks.append(int(j))
            picks = picks[:k]
            mean, std, ei = m[picks], s[picks], e[picks]
        else:
            # Ablation: no surrogate. Present a random shortlist, descriptions only.
            picks = list(rng.choice(len(unseen), size=k, replace=False))

        cand_ids = [int(unseen[j]) for j in picks]
        ctx = DecisionContext(
            iteration=iteration, budget=budget, remaining=budget - iteration + 1,
            objective_label=dataset.objective_label, legend=dataset.legend,
            hist_desc=[dataset.descriptions[i] for i in evaluated],
            y_eval=np.asarray(y_obs), X_eval=dataset.X[evaluated],
            cand_desc=[dataset.descriptions[i] for i in cand_ids], cand_ids=cand_ids,
            X_cand=dataset.X[cand_ids], mean=mean, std=std, ei=ei,
            use_surrogate=self.use_surrogate, n_select=min(q, len(cand_ids)),
        )
        positions = self.agent.select_batch(ctx)
        self._log_decision(dataset, seed, ctx, positions)
        return [cand_ids[p] for p in positions]

    def _log_decision(self, dataset, seed, ctx, positions):
        """Append an audit record of this decision to the (shared) agent's decision_log.

        Records the full shortlist with ground truth alongside the agent's stated
        strategy/rationale, so scripts/audit_decisions.py can check the reasoning
        against reality after the run — no API calls needed.
        """
        log = getattr(self.agent, "decision_log", None)
        if log is None:
            log = self.agent.decision_log = []
        shortlist = []
        for pos, gid in enumerate(ctx.cand_ids):
            entry = {"id": int(gid), "desc": ctx.cand_desc[pos], "y": float(dataset.y[gid])}
            if dataset.y_true is not None:
                entry["y_true"] = float(dataset.y_true[gid])
            if ctx.mean is not None:
                entry.update(mean=float(ctx.mean[pos]), std=float(ctx.std[pos]),
                             ei=float(ctx.ei[pos]))
            shortlist.append(entry)
        dec = getattr(self.agent, "last_decision", None) or {}
        log.append({
            "dataset": dataset.title, "policy": self.name, "seed": int(seed),
            "round": int(ctx.iteration), "n_rounds": int(ctx.budget),
            "best_so_far": ctx.best_value, "n_select": int(ctx.n_select),
            "strategy": dec.get("strategy"), "rationale": dec.get("rationale"),
            "fallback": bool(dec.get("fallback", False)),
            "cache_hit": bool(dec.get("cache_hit", False)),
            "picked_pos": [int(p) for p in positions],
            "shortlist": shortlist,
        })


def _length_scale(Xc, rng, sample: int = 200) -> float:
    """Median pairwise distance over a subsample — the penalisation radius for batching."""
    m = len(Xc)
    if m <= 1:
        return 1.0
    idx = rng.choice(m, size=min(sample, m), replace=False)
    S = Xc[idx]
    d = np.linalg.norm(S[:, None, :] - S[None, :, :], axis=-1)
    d = d[np.triu_indices(len(S), k=1)]
    d = d[d > 0]
    return float(np.median(d)) if d.size else 1.0


def _greedy_batch_lp(Xc, acq, q: int, ell: float) -> list:
    """Greedy batch selection with local penalization around already-chosen points."""
    pen = acq.astype(float).copy()
    picks: list[int] = []
    for _ in range(q):
        j = int(np.argmax(pen))
        picks.append(j)
        d = np.linalg.norm(Xc - Xc[j], axis=1)
        pen = pen * (1.0 - np.exp(-((d / max(ell, 1e-9)) ** 2)))  # suppress neighbours of j
        for p in picks:
            pen[p] = -np.inf
    return picks


def _unseen(dataset: Dataset, evaluated) -> np.ndarray:
    mask = np.ones(dataset.n, dtype=bool)
    mask[np.asarray(evaluated, dtype=int)] = False
    return np.flatnonzero(mask)


def _fit_predict(dataset, evaluated, y_obs, query_idx, seed):
    surrogate = GaussianSurrogate(seed=seed).fit(dataset.X[evaluated], np.asarray(y_obs))
    return surrogate.predict(dataset.X[query_idx])
