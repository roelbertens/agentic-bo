"""Agentic Bayesian optimisation: LLM agents vs classic BO on pool-based
optimisation tasks (synthetic MOF landscape + two real reaction datasets).
Data loaders live in the shared ``datasets`` package; see ``run_agentic_bo.py``
and docs/AGENTIC_BO.md."""

__all__ = ["surrogate", "agent", "agentic", "policies", "experiment", "plotting",
           "cache", "tracing"]
