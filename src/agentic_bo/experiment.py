"""Run optimisation policies over multiple random seeds and collect convergence."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from datasets import Dataset


@dataclass
class MethodResult:
    name: str
    curves: np.ndarray                    # (n_runs, n_init + budget) best-so-far per evaluation
    evaluations: np.ndarray               # (n_init + budget,) x-axis: cumulative evaluations
    best_possible: float
    round_quality: np.ndarray | None = None  # (n_runs, n_rounds) per-round proposal quality (IMP@k)

    @property
    def mean(self) -> np.ndarray:
        return self.curves.mean(axis=0)

    @property
    def std(self) -> np.ndarray:
        return self.curves.std(axis=0)

    @property
    def final_regret(self) -> np.ndarray:
        """Simple regret (best_possible - best_found) at the end of each run."""
        return self.best_possible - self.curves[:, -1]


def run_single(dataset: Dataset, policy, seed: int, budget: int, n_init: int,
               batch_size: int = 1, return_rounds: bool = False):
    """One optimisation run.

    ``budget`` is the number of experiments after initialisation; with
    ``batch_size`` q, they are proposed q per round (matching batch-BO protocols).
    Returns the best-so-far curve over all evaluations. If ``return_rounds`` is set,
    also returns the *per-round proposal quality* (best yield among each round's
    proposed batch) — a non-cumulative signal, matching Reasoning-BO's IMP@k metric.
    """
    rng = np.random.default_rng(seed)
    init = rng.choice(dataset.n, size=n_init, replace=False)
    order = list(int(i) for i in init)
    y_obs = list(dataset.y[order])

    total_rounds = int(np.ceil(budget / batch_size))
    round_quality = []
    for r in range(1, total_rounds + 1):
        q = min(batch_size, budget - (r - 1) * batch_size, dataset.n - len(order))
        if q <= 0:
            break
        picks = policy.propose_batch(dataset, order, y_obs, iteration=r,
                                     budget=total_rounds, rng=rng, seed=seed, q=q)
        round_quality.append(float(max(dataset.y[i] for i in picks)))
        for idx in picks:
            order.append(int(idx))
            y_obs.append(float(dataset.y[idx]))

    curve = np.maximum.accumulate(np.asarray(y_obs))
    return (curve, np.asarray(round_quality)) if return_rounds else curve


def run_method(
    dataset: Dataset, make_policy, name: str, seeds, budget: int, n_init: int,
    batch_size: int = 1, verbose: bool = True
) -> MethodResult:
    curves, rounds = [], []
    for seed in seeds:
        if verbose:
            print(f"  {name}: seed {seed} ...", flush=True)
        c, rq = run_single(dataset, make_policy(), seed, budget, n_init, batch_size,
                           return_rounds=True)
        curves.append(c)
        rounds.append(rq)
    evaluations = np.arange(1, n_init + budget + 1)
    return MethodResult(name=name, curves=np.vstack(curves), evaluations=evaluations,
                        best_possible=dataset.best_value, round_quality=np.vstack(rounds))


def run_method_per_substrate(
    datasets, make_policy, name: str, seeds, budget: int, n_init: int,
    batch_size: int = 1, verbose: bool = True
) -> MethodResult:
    """Run one optimisation per (substrate, seed), normalised to % of each substrate's optimum.

    Averaging over substrates only makes sense once each run is scaled by its own
    optimum, since substrates have different maximum yields.
    """
    curves, rounds = [], []
    for d, ds in enumerate(datasets):
        opt = ds.best_value
        for seed in seeds:
            if verbose:
                print(f"  {name}: substrate {d} seed {seed} ...", flush=True)
            curve, rq = run_single(ds, make_policy(), seed, budget, n_init, batch_size,
                                   return_rounds=True)
            curves.append(100.0 * curve / opt)          # % of this substrate's optimum
            rounds.append(100.0 * rq / opt)
    evaluations = np.arange(1, n_init + budget + 1)
    return MethodResult(name=name, curves=np.vstack(curves), evaluations=evaluations,
                        best_possible=100.0, round_quality=np.vstack(rounds))
