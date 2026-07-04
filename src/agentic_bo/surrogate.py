"""Gaussian-process surrogate and the Expected-Improvement acquisition function.

This is the classic Bayesian-optimisation machinery: fit a GP to the observed
(descriptor, capacity) pairs, then score unseen candidates by how much
improvement over the current best we can expect from measuring them.
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
