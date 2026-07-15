"""Campaign state and the multi-seed experiment runner.

A campaign is one budgeted optimisation run: a policy spends the cost budget on
``simulate``/``measure`` calls through :class:`CampaignState`, and everything it
did is recovered from the ledger afterwards. Policies expose a single
``run(state, rng)`` method and are free to spend the budget however they like —
which is the point of the experiment.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .oracles import (
    HIGH,
    LOW,
    CostLedger,
    MultiFidelityProblem,
    best_hf_curve,
    hf_share_curve,
)


class CampaignState:
    """What a policy sees and does: the two oracles, the ledger, and the history."""

    def __init__(self, problem: MultiFidelityProblem, budget: float):
        self.problem = problem
        self.ledger = CostLedger(budget)
        self.lf_obs: dict[int, float] = {}   # candidate -> simulated value
        self.hf_obs: dict[int, float] = {}   # candidate -> measured value

    # --- oracle access -------------------------------------------------------------
    def simulate(self, candidates) -> list[float]:
        out = []
        for c in candidates:
            self.lf_obs[int(c)] = v = self.ledger.charge(self.problem, LOW, int(c))
            out.append(v)
        return out

    def measure(self, candidates) -> list[float]:
        out = []
        for c in candidates:
            self.hf_obs[int(c)] = v = self.ledger.charge(self.problem, HIGH, int(c))
            out.append(v)
        return out

    # --- bookkeeping the policies need ----------------------------------------------
    @property
    def remaining(self) -> float:
        return self.ledger.remaining

    def can_measure(self) -> bool:
        return self.ledger.can_afford(self.problem.cost_hf)

    def can_simulate(self) -> bool:
        return self.ledger.can_afford(self.problem.cost_lf)

    @property
    def best_measured(self) -> float | None:
        return max(self.hf_obs.values()) if self.hf_obs else None

    def unmeasured(self) -> np.ndarray:
        return np.setdiff1d(np.arange(self.problem.n), list(self.hf_obs))

    def unsimulated(self) -> np.ndarray:
        return np.setdiff1d(np.arange(self.problem.n), list(self.lf_obs))


@dataclass
class MethodResult:
    name: str
    curves: np.ndarray       # (n_runs, budget + 1) best measured value per unit cost
    hf_share: np.ndarray     # (n_runs, budget + 1) fraction of spend on measurements
    best_possible: float

    @property
    def mean(self) -> np.ndarray:
        return np.nanmean(self.curves, axis=0)

    @property
    def std(self) -> np.ndarray:
        return np.nanstd(self.curves, axis=0)

    @property
    def final_pct(self) -> np.ndarray:
        """Final best measurement per run, as % of the pool optimum."""
        return 100.0 * self.curves[:, -1] / self.best_possible


def run_campaign(problem: MultiFidelityProblem, policy, seed: int, budget: float,
                 n_init: int) -> CampaignState:
    """One campaign: shared random init (``n_init`` measurements, charged like any
    other spend), then the policy takes over until it stops or runs dry."""
    rng = np.random.default_rng(seed)
    state = CampaignState(problem, budget)
    init = rng.choice(problem.n, size=n_init, replace=False)
    state.measure(int(i) for i in init)
    policy.run(state, rng)
    return state


def run_method(problem: MultiFidelityProblem, make_policy, name: str, seeds,
               budget: float, n_init: int, verbose: bool = True) -> MethodResult:
    curves, shares = [], []
    for seed in seeds:
        if verbose:
            print(f"  {name}: seed {seed} ...", flush=True)
        state = run_campaign(problem, make_policy(), seed, budget, n_init)
        curves.append(best_hf_curve(state.ledger.log, budget))
        shares.append(hf_share_curve(state.ledger.log, problem, budget))
    return MethodResult(name=name, curves=np.vstack(curves), hf_share=np.vstack(shares),
                        best_possible=problem.dataset.best_value)
