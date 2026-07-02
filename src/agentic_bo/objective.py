"""Physically-motivated synthetic MOF dataset for CO2-capture optimisation.

Each "material" (a metal-organic framework) is described by five normalised
descriptors. The objective is a smooth, mildly multi-modal function that rewards
high surface area, an *intermediate* pore size, high void fraction, and
favourable metal chemistry — the qualitative trends seen in real CO2-capture MOF
screens. The landscape is deterministic, so the global optimum over the pool is
known exactly and *simple regret* is well defined.

This is the easy, smooth benchmark: classic GP+EI is near-optimal here, so it is
useful mainly as a sanity check and a floor for how much an agent can add. For a
setting where an LLM agent's prior knowledge can actually help, see
``reactions.py`` (a real, discrete reaction-optimisation dataset).
"""
from __future__ import annotations

import numpy as np

from .data import Dataset

# name, physical lower bound, physical upper bound, unit
DESCRIPTORS = [
    ("surface_area", 500.0, 6000.0, "m^2/g"),
    ("pore_diameter", 3.0, 30.0, "Angstrom"),
    ("void_fraction", 0.20, 0.90, "-"),
    ("metal_affinity", 0.0, 1.0, "-"),
    ("linker_length", 5.0, 25.0, "Angstrom"),
]
_NAMES = [d[0] for d in DESCRIPTORS]
_UNITS = [d[3] for d in DESCRIPTORS]
_LO = np.array([d[1] for d in DESCRIPTORS])
_HI = np.array([d[2] for d in DESCRIPTORS])


def _capacity(u: np.ndarray) -> np.ndarray:
    """CO2 working capacity (mol/kg) as a function of normalised descriptors u in [0,1]^5."""
    sa, pore, void, metal, link = (u[:, i] for i in range(5))
    return (
        3.5 * sa                                        # more surface area helps ...
        - 1.0 * sa * pore                               # ... but huge pores waste it
        + 4.0 * np.exp(-((pore - 0.45) / 0.18) ** 2)    # optimum at intermediate pore size
        + 2.5 * void                                    # accessible volume
        + 3.0 * metal * void                            # metal chemistry x accessible volume
        + 1.5 * np.exp(-((link - 0.60) / 0.25) ** 2)    # mild linker-length optimum
        + 4.0                                           # offset -> keep values positive
    )


def _describe(x_phys: np.ndarray) -> str:
    return ", ".join(f"{n}={v:.1f}{('' if u == '-' else ' ' + u)}"
                     for n, v, u in zip(_NAMES, x_phys, _UNITS))


def sample_pool(n: int = 600, seed: int = 0) -> Dataset:
    """Sample a reproducible pool of candidate MOFs.

    The pool seed is intentionally separate from the optimisation seed so that
    every optimiser searches the *same* landscape across runs.
    """
    rng = np.random.default_rng(seed)
    X = rng.random((n, len(DESCRIPTORS)))
    y = _capacity(X)
    phys = _LO + X * (_HI - _LO)
    descriptions = [_describe(phys[i]) for i in range(n)]
    return Dataset(
        X=X, y=y, descriptions=descriptions,
        objective_label="CO2 working capacity (mol/kg)",
        title="MOF CO2 capture (synthetic)",
        legend="Descriptors: " + ", ".join(f"{n} ({u})" for n, u in zip(_NAMES, _UNITS)),
    )
