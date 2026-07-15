"""Surrogates and acquisition for the plan-pool BO baselines of the design task.

The GP + Expected Improvement pair mirrors the machinery in the agentic-BO study
(each experiment keeps its own copy; only the data loaders are shared). The
Bayesian linear model is specific to this study: on the additive count features
of a chain (``design.chain_features``) the reward *is* linear, so its posterior
mean coefficients are the reward model a planner can act on.
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


class BayesianLinear:
    """Bayesian linear-regression surrogate with a closed-form posterior.

    Same ``fit``/``predict`` interface as :class:`GaussianSurrogate`, so it drops
    into the same BO loop. Use it when the objective is (near-)linear in engineered
    features — e.g. the additive block/adjacency counts of a design
    (``design.chain_features``) — where it is a far stronger, better-identified
    surrogate than a generic GP over raw one-hots. ``weights`` exposes the posterior
    mean coefficients, which for those features *are* the reward model (per-block and
    per-adjacency terms), so a planner can act on them.
    """

    def __init__(self, alpha: float = 1e-2, beta: float = 1.0):
        self.alpha = alpha              # prior precision on the weights
        self.beta = beta                # observation-noise precision

    def fit(self, X: np.ndarray, y: np.ndarray) -> BayesianLinear:
        y = np.asarray(y, dtype=float)
        d = X.shape[1]
        self.cov = np.linalg.inv(self.alpha * np.eye(d) + self.beta * X.T @ X)
        self.w = self.beta * self.cov @ X.T @ y
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = X @ self.w
        var = 1.0 / self.beta + np.einsum("ij,ij->i", X @ self.cov, X)
        return mean, np.sqrt(np.maximum(var, 1e-12))

    def weights(self) -> np.ndarray:
        return self.w


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
