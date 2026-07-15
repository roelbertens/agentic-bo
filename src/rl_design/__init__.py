"""Sequential linker design: RL vs. planning vs. BO on a genuine episodic MDP
(build a chain block by block under stochastic couplings). Self-contained;
see ``run_rl_design.py`` and docs/RL_DESIGN.md."""

__all__ = ["design", "reinforce", "methods", "surrogate"]
