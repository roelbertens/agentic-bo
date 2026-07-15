"""Two oracles over one candidate pool, and the cost ledger that pays for them.

The high-fidelity oracle ("measure") returns the pool's true objective at cost
``cost_hf``. The low-fidelity oracle ("simulate") returns a *corrupted proxy* at
cost ``cost_lf``: the true values are standardised, mixed with an independent
Gaussian landscape, and rescaled, so that the LF/HF correlation is the knob
``rho``. The corruption is drawn once per problem (seeded), which makes the
simulator deterministic — querying the same candidate twice returns the same
number and buys no new information, exactly like a systematically-wrong
simulator and unlike a merely noisy one.

No real cheap simulator exists for these datasets, so the LF oracle is
constructed — standard practice in multi-fidelity benchmarks. Making ``rho``
explicit turns that limitation into the experiment's x-axis: *how good does the
simulator have to be before spending on it pays off?*
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from datasets import Dataset

LOW, HIGH = "lf", "hf"


class BudgetExceeded(RuntimeError):
    """Raised when an oracle call would overspend the campaign's cost budget."""


@dataclass(frozen=True)
class Observation:
    """One oracle call, as recorded in the ledger (cost is cumulative *after* it)."""

    candidate: int
    fidelity: str          # LOW or HIGH
    value: float
    cost_after: float


class MultiFidelityProblem:
    """A dataset pool wrapped with a cheap corrupted oracle next to the true one."""

    def __init__(self, dataset: Dataset, rho: float = 0.8,
                 cost_lf: float = 1.0, cost_hf: float = 10.0, seed: int = 0):
        assert 0.0 <= rho <= 1.0, "rho is a correlation, in [0, 1]"
        assert cost_lf < cost_hf, "a simulation must be cheaper than a measurement"
        self.dataset = dataset
        self.rho = float(rho)
        self.cost_lf = float(cost_lf)
        self.cost_hf = float(cost_hf)
        y = dataset.y
        mu, sd = float(y.mean()), float(y.std() + 1e-12)
        z = (y - mu) / sd
        # One fixed independent landscape per problem: rho mixes truth and distractor,
        # so corr(y_lf, y) -> rho over the pool (exact in expectation, seeded here).
        g = np.random.default_rng(seed).normal(size=dataset.n)
        self.y_lf = mu + sd * (self.rho * z + np.sqrt(1.0 - self.rho ** 2) * g)

    @property
    def n(self) -> int:
        return self.dataset.n

    @property
    def cost_ratio(self) -> float:
        return self.cost_hf / self.cost_lf

    def cost(self, fidelity: str) -> float:
        return self.cost_lf if fidelity == LOW else self.cost_hf

    def query(self, fidelity: str, candidate: int) -> float:
        src = self.y_lf if fidelity == LOW else self.dataset.y
        return float(src[candidate])

    def empirical_rho(self) -> float:
        """Realised pool-wide LF/HF correlation (close to ``rho`` for large pools)."""
        return float(np.corrcoef(self.y_lf, self.dataset.y)[0, 1])


class CostLedger:
    """Single source of truth for what a campaign did and what it paid.

    Every oracle call goes through :meth:`charge`; convergence curves and the
    fidelity mix are derived from ``log`` afterwards, so a policy cannot report
    progress it did not pay for.
    """

    def __init__(self, budget: float):
        self.budget = float(budget)
        self.spent = 0.0
        self.log: list[Observation] = []

    @property
    def remaining(self) -> float:
        return self.budget - self.spent

    def can_afford(self, cost: float) -> bool:
        return self.spent + cost <= self.budget + 1e-9

    def charge(self, problem: MultiFidelityProblem, fidelity: str, candidate: int) -> float:
        cost = problem.cost(fidelity)
        if not self.can_afford(cost):
            raise BudgetExceeded(
                f"{fidelity} query costs {cost:g}, only {self.remaining:g} of "
                f"{self.budget:g} left")
        self.spent += cost
        value = problem.query(fidelity, candidate)
        self.log.append(Observation(int(candidate), fidelity, value, self.spent))
        return value


def best_hf_curve(log: list[Observation], budget: float) -> np.ndarray:
    """Best *measured* value so far, on the integer cost grid 0..budget.

    Only high-fidelity observations count — a simulated value is a claim, not a
    result. Grid points before the first measurement are NaN.
    """
    grid_len = int(np.floor(budget)) + 1
    curve = np.full(grid_len, np.nan)
    best = np.nan
    prev = 0
    for obs in log:
        idx = min(int(np.ceil(obs.cost_after)), grid_len - 1)
        curve[prev:idx] = best
        if obs.fidelity == HIGH:
            best = obs.value if np.isnan(best) else max(best, obs.value)
        prev = idx
    curve[prev:] = best
    return curve


def hf_share_curve(log: list[Observation], problem: MultiFidelityProblem,
                   budget: float) -> np.ndarray:
    """Fraction of the money spent so far that went to measurements, on the same
    grid — the visible trace of *when* a method trusts the simulator."""
    grid_len = int(np.floor(budget)) + 1
    share = np.full(grid_len, np.nan)
    spent_hf = 0.0
    current = np.nan
    prev = 0
    for obs in log:
        idx = min(int(np.ceil(obs.cost_after)), grid_len - 1)
        share[prev:idx] = current
        if obs.fidelity == HIGH:
            spent_hf += problem.cost_hf
        current = spent_hf / obs.cost_after
        prev = idx
    share[prev:] = current
    return share
