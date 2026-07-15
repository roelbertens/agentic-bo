"""GP surrogate, Expected Improvement, and the augmented-input multi-fidelity fit.

The GP + EI pair mirrors the other studies' machinery (each experiment keeps its
own copy; only the data loaders are shared). The multi-fidelity twist is the
simplest one that works: append a fidelity flag to the features and fit one GP
on simulations and measurements together, so the kernel *learns from the data*
how much the cheap oracle resembles the expensive one instead of being told.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel


class GaussianSurrogate:
    """A thin wrapper around scikit-learn's GP with a Matern 5/2 kernel."""

    def __init__(self, seed: int = 0):
        self.seed = seed
        self.gp: GaussianProcessRegressor | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> GaussianSurrogate:
        kernel = (
            ConstantKernel(1.0, (1e-2, 1e2))
            * Matern(length_scale=0.3, length_scale_bounds=(1e-2, 1e1), nu=2.5)
            + WhiteKernel(1e-2, (1e-3, 1e0))  # keep noise off the lower bound -> stable Cholesky
        )
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=2,
            alpha=1e-6,  # jitter on the diagonal for numerical stability
            random_state=self.seed,
        )
        self.gp.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return posterior (mean, std) at the query points."""
        assert self.gp is not None, "call fit() before predict()"
        # A near-interpolating GP can make the predictive covariance ill-conditioned;
        # ignore the resulting harmless numpy warnings, values remain well-defined.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            mean, std = self.gp.predict(X, return_std=True)
        return mean, std


def expected_improvement(
    mean: np.ndarray, std: np.ndarray, best: float, xi: float = 0.01
) -> np.ndarray:
    """Expected Improvement for a maximisation problem.

    xi trades off exploration vs exploitation; a small positive value nudges the
    search away from points that only marginally beat the incumbent.
    """
    std = np.maximum(std, 1e-9)
    improvement = mean - best - xi
    z = improvement / std
    ei = improvement * norm.cdf(z) + std * norm.pdf(z)
    return np.maximum(ei, 0.0)


def _augment(X: np.ndarray, flag: float) -> np.ndarray:
    """Append the fidelity flag (0 = simulation, 1 = measurement) as a feature."""
    return np.hstack([X, np.full((len(X), 1), flag)])


class MultiFidelitySurrogate:
    """One GP over (features, fidelity flag), fit on both kinds of observation.

    ``predict_hf``/``predict_lf`` give the posterior at either fidelity for any
    candidate. When the simulator is informative the LF points sharpen the HF
    posterior; when it is junk the kernel learns to decorrelate the two levels
    (the flag dimension gets a short length-scale) and the LF points stop
    mattering — the desired behaviour, learned rather than assumed.
    """

    def __init__(self, X_pool: np.ndarray, seed: int = 0):
        self.X_pool = X_pool
        self.gp = GaussianSurrogate(seed=seed)

    def fit(self, lf_obs: dict[int, float], hf_obs: dict[int, float]) -> MultiFidelitySurrogate:
        rows, targets = [], []
        for flag, obs in ((0.0, lf_obs), (1.0, hf_obs)):
            if obs:
                ids = np.fromiter(obs, dtype=int)
                rows.append(_augment(self.X_pool[ids], flag))
                targets.append(np.array([obs[int(i)] for i in ids]))
        self.gp.fit(np.vstack(rows), np.concatenate(targets))
        return self

    def predict_hf(self, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.gp.predict(_augment(self.X_pool[np.asarray(ids, dtype=int)], 1.0))

    def predict_lf(self, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.gp.predict(_augment(self.X_pool[np.asarray(ids, dtype=int)], 0.0))
