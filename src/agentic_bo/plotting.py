"""Convergence plots: best-so-far and simple regret, mean +/- std across seeds."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless / no display
import matplotlib.pyplot as plt
import numpy as np

_COLORS = {
    "random": "#9e9e9e",
    "classic_bo": "#1f77b4",
    "agentic_bo": "#d62728",
    "agentic_no_surrogate": "#ff7f0e",
}
_LABELS = {
    "random": "Random search",
    "classic_bo": "Classic BO (GP + EI)  [baseline]",
    "agentic_bo": "Agentic BO (agent + surrogate)",
    "agentic_no_surrogate": "Agentic BO, no surrogate  [ablation]",
}


def plot_convergence(results, agent_name: str, n_init: int, out_path: str,
                     objective_label: str = "objective", title: str = "") -> None:
    best = results[0].best_possible
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for res in results:
        color = _COLORS.get(res.name)
        label = _LABELS.get(res.name, res.name)
        x = res.evaluations
        ax1.plot(x, res.mean, color=color, label=label, lw=2)
        ax1.fill_between(x, res.mean - res.std, res.mean + res.std, color=color, alpha=0.15)

        regret = np.maximum(best - res.curves, 1e-6)
        ax2.plot(x, regret.mean(axis=0), color=color, label=label, lw=2)

    ax1.axhline(best, ls="--", color="k", lw=1, alpha=0.6, label="Global optimum (pool)")
    ax1.axvline(n_init, ls=":", color="k", lw=1, alpha=0.4)
    ax1.set_xlabel("Candidates evaluated")
    ax1.set_ylabel(f"Best found: {objective_label}")
    ax1.set_title("Convergence  (mean +/- std over seeds)")
    ax1.legend(loc="lower right", fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.set_yscale("log")
    ax2.axvline(n_init, ls=":", color="k", lw=1, alpha=0.4)
    ax2.set_xlabel("Candidates evaluated")
    ax2.set_ylabel("Simple regret (log scale)")
    ax2.set_title("Regret to global optimum")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(alpha=0.3, which="both")

    fig.suptitle(
        f"Agentic Bayesian Optimisation — {title}  (agent: {agent_name})",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
