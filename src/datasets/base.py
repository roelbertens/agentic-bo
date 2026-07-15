"""Dataset container shared by every optimisation task.

A task is just a finite pool of candidates, each with numeric features (for the
GP surrogate), a scalar objective to *maximise*, and a human-readable
description (for the LLM agent to reason over). Keeping this dataset-agnostic lets
the exact same policies run on the synthetic MOF landscape and on a real
reaction-optimisation dataset.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass
class Dataset:
    X: np.ndarray                 # (n, d) numeric features for the GP surrogate
    y: np.ndarray                 # (n,) objective value (maximisation)
    descriptions: list            # (n,) human-readable label per candidate (for the LLM)
    objective_label: str          # e.g. "CO2 working capacity (mol/kg)"
    title: str = "dataset"        # short name for plot titles
    legend: str = ""              # optional shared context block for the LLM prompt
    y_true: np.ndarray | None = None  # original objective when y has been permuted (leakage check)

    @property
    def n(self) -> int:
        return self.X.shape[0]

    @property
    def dim(self) -> int:
        return self.X.shape[1]

    @property
    def best_value(self) -> float:
        return float(self.y.max())


def permute_yields(dataset: Dataset, seed: int = 0) -> Dataset:
    """Return a copy with the objective values shuffled across candidates.

    Descriptions and features stay put, so any chemistry knowledge the agent
    brings no longer maps to the observed objective. The original values are
    kept in ``y_true`` so an audit can test whether the agent keeps chasing
    the *original* optimum — evidence of dataset memorisation, not reasoning.
    """
    rng = np.random.default_rng(seed)
    return replace(dataset, y=dataset.y[rng.permutation(dataset.n)], y_true=dataset.y,
                   title=dataset.title + " [yields permuted]")
