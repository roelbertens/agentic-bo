"""Multi-fidelity experiment: when does a cheap simulator pay off, and can an
agent learn to trust or distrust it? Two oracles per candidate — a cheap,
corrupted simulation and an expensive ground-truth measurement — with the budget
in cost units. Self-contained apart from the shared ``datasets`` loaders; see
``run_multi_fidelity.py`` and docs/MULTI_FIDELITY.md."""

__all__ = ["oracles", "campaign", "surrogate", "baselines", "agent", "heuristic", "cache"]
