"""The agentic policy: a PydanticAI agent that spends the budget itself.

Unlike the baselines, nothing here hard-codes *when* to simulate versus measure.
The agent gets five typed tools — inspect the campaign (``recall``), rank
candidates under the multi-fidelity surrogate (``shortlist``), query that
surrogate on candidates of its own choosing (``predict``, free), and the two
oracles (``simulate``/``measure``, which really spend budget) — plus the standing
instruction that the simulator's reliability is unknown and must be inferred
from the LF/HF pairs it has produced. Tool argument validation is PydanticAI's:
an invalid id, a re-query, or an unaffordable call raises ``ModelRetry`` and the
model gets the error text back to correct itself.

The campaign runs in rounds. Each round is one ``agent.run_sync`` ending in a
structured :class:`RoundDecision` (rationale, a scratchpad note carried to the
next round, and an optional stop). Budget, not round count, ends the campaign.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.messages import RetryPromptPart
from pydantic_ai.usage import UsageLimits

from .campaign import CampaignState
from .oracles import HIGH
from .surrogate import MultiFidelitySurrogate, expected_improvement

# Pinned by a golden test: the replay cache is keyed by the exact request, so an
# accidental wording change silently turns free cache hits into new API calls.
INSTRUCTIONS = (
    "You run a budgeted experimental campaign to find the candidate with the highest "
    "objective value in a fixed pool. You have two oracles: simulate(candidate_ids) is a "
    "cheap computational proxy, measure(candidate_ids) is the expensive ground truth. "
    "Only measured values count as results. The simulator is deterministic (re-querying "
    "a candidate returns the same number and wastes budget) and its reliability is "
    "UNKNOWN: it may track the truth closely or be badly misleading. Infer how much to "
    "trust it from candidates where you have both a simulation and a measurement, and "
    "adapt your mix: a trustworthy simulator lets you screen broadly before confirming, "
    "a misleading one should be abandoned for direct measurement. Spend the whole budget "
    "unless further spending cannot improve the best measurement."
)

ROUND_PROMPT = (
    "Round {round}. Spent {spent:g} of {budget:g}; remaining {remaining:g}. "
    "Costs per candidate: simulate {cost_lf:g}, measure {cost_hf:g}.\n"
    "Best measured so far: {best:.2f}.\n"
    "Notes from earlier rounds:\n{memory}\n\n"
    "Investigate with recall/shortlist/predict, then spend budget with simulate/measure. "
    "Finish with your rationale, a short note to your future self, and stop=true only "
    "if further spending cannot improve the best measurement."
)


class RoundDecision(BaseModel):
    """The structured end-of-round output the agent must produce."""

    rationale: str
    note: str = ""     # scratchpad line carried into the next round's prompt
    stop: bool = False


@dataclass
class AgentDeps:
    state: CampaignState
    shortlist_k: int = 8


def _check_ids(ids: list[int], n: int) -> list[int]:
    bad = [i for i in ids if not 0 <= i < n]
    if bad or not ids:
        raise ModelRetry(f"invalid candidate ids {bad or ids}; the pool is 0..{n - 1}")
    if len(set(ids)) != len(ids):
        raise ModelRetry("duplicate ids in one call — each candidate at most once")
    return [int(i) for i in ids]


def build_agent(model) -> Agent[AgentDeps, RoundDecision]:
    agent: Agent[AgentDeps, RoundDecision] = Agent(
        model, deps_type=AgentDeps, output_type=RoundDecision,
        instructions=INSTRUCTIONS, retries=3)

    @agent.tool
    def recall(ctx: RunContext[AgentDeps]) -> str:
        """Campaign state: budget, all measurements, simulator-vs-truth pairs, and
        the top simulated-but-unmeasured candidates."""
        s = ctx.deps.state
        p = s.problem
        lines = [f"Budget: remaining={s.remaining:g} spent={s.ledger.spent:g} "
                 f"cost_lf={p.cost_lf:g} cost_hf={p.cost_hf:g}"]
        measured = sorted(s.hf_obs.items(), key=lambda kv: -kv[1])
        lines.append(f"Measured ({len(measured)}):")
        lines += [f"  [id={c}] {p.dataset.descriptions[c]} -> {v:.2f}" for c, v in measured[:30]]
        pairs = [(c, s.lf_obs[c], v) for c, v in measured if c in s.lf_obs]
        lines.append(f"Simulator vs truth on {len(pairs)} shared candidate(s):")
        lines += [f"  [id={c}] sim={lf:.2f} true={v:.2f} error={lf - v:+.2f}"
                  for c, lf, v in pairs[:20]]
        unconfirmed = sorted(((c, v) for c, v in s.lf_obs.items() if c not in s.hf_obs),
                             key=lambda kv: -kv[1])
        lines.append(f"Simulated only ({len(unconfirmed)}), top by simulated value:")
        lines += [f"  [id={c}] sim={v:.2f}" for c, v in unconfirmed[:15]]
        return "\n".join(lines)

    @agent.tool
    def shortlist(ctx: RunContext[AgentDeps], k: int = 0) -> str:
        """Rank unmeasured candidates under a surrogate fit to ALL observations
        (simulations and measurements): top expected improvement plus the most
        uncertain, with any known simulated value shown per candidate."""
        s = ctx.deps.state
        k = k or ctx.deps.shortlist_k
        um = s.unmeasured()
        if not len(um):
            return "(everything is measured)"
        surr = MultiFidelitySurrogate(s.problem.dataset.X, seed=0)
        surr.fit(s.lf_obs, s.hf_obs)
        mean, std = surr.predict_hf(um)
        ei = expected_improvement(mean, std, best=s.best_measured)
        n_explore = min(2, k)
        picks = list(np.argsort(-ei)[: k - n_explore])
        picks += [int(j) for j in np.argsort(-std) if int(j) not in picks][:k - len(picks)]
        lines = []
        for j in picks:
            c = int(um[j])
            sim = f"{s.lf_obs[c]:.2f}" if c in s.lf_obs else "none"
            lines.append(f"[id={c}] EI={ei[j]:.3f} pred={mean[j]:.2f}+/-{std[j]:.2f} "
                         f"sim={sim} | {s.problem.dataset.descriptions[c]}")
        return "\n".join(lines)

    @agent.tool
    def predict(ctx: RunContext[AgentDeps], candidate_ids: list[int]) -> str:
        """Free surrogate predictions for specific candidates (model fit to all
        observations): measurement-fidelity mean +/- std and EI, with any known
        simulated or measured value alongside. Costs no budget."""
        s = ctx.deps.state
        ids = _check_ids(candidate_ids, s.problem.n)
        surr = MultiFidelitySurrogate(s.problem.dataset.X, seed=0)
        surr.fit(s.lf_obs, s.hf_obs)
        mean, std = surr.predict_hf(np.array(ids))
        ei = expected_improvement(mean, std, best=s.best_measured)
        lines = []
        for j, c in enumerate(ids):
            known = (f"measured={s.hf_obs[c]:.2f}" if c in s.hf_obs
                     else f"sim={s.lf_obs[c]:.2f}" if c in s.lf_obs else "unqueried")
            lines.append(f"[id={c}] pred={mean[j]:.2f}+/-{std[j]:.2f} EI={ei[j]:.3f} "
                         f"{known} | {s.problem.dataset.descriptions[c]}")
        return "\n".join(lines)

    @agent.tool
    def simulate(ctx: RunContext[AgentDeps], candidate_ids: list[int]) -> str:
        """Run the cheap simulator on candidates (spends cost_lf each)."""
        s = ctx.deps.state
        ids = _check_ids(candidate_ids, s.problem.n)
        known = [i for i in ids if i in s.lf_obs]
        if known:
            raise ModelRetry(f"already simulated (deterministic, no new information): {known}")
        total = len(ids) * s.problem.cost_lf
        if not s.ledger.can_afford(total):
            raise ModelRetry(f"simulating {len(ids)} costs {total:g} but only "
                             f"{s.remaining:g} remains — fewer candidates, or stop")
        values = s.simulate(ids)
        body = "\n".join(f"[id={c}] -> {v:.2f}" for c, v in zip(ids, values, strict=True))
        return f"{body}\nremaining={s.remaining:g}"

    @agent.tool
    def measure(ctx: RunContext[AgentDeps], candidate_ids: list[int]) -> str:
        """Measure ground truth on candidates (spends cost_hf each) — the only
        values that count as results."""
        s = ctx.deps.state
        ids = _check_ids(candidate_ids, s.problem.n)
        known = [i for i in ids if i in s.hf_obs]
        if known:
            raise ModelRetry(f"already measured: {known}")
        total = len(ids) * s.problem.cost_hf
        if not s.ledger.can_afford(total):
            raise ModelRetry(f"measuring {len(ids)} costs {total:g} but only "
                             f"{s.remaining:g} remains — fewer candidates, or stop")
        values = s.measure(ids)
        body = "\n".join(f"[id={c}] -> {v:.2f}" for c, v in zip(ids, values, strict=True))
        return f"{body}\nremaining={s.remaining:g}"

    return agent


class AgenticMF:
    """Campaign policy driven by a PydanticAI agent (same ``run`` interface as the
    baselines). ``model`` is anything PydanticAI accepts — a model name string, a
    ``FunctionModel`` (the offline heuristic), or a cache-wrapped model."""

    name = "agentic_mf"

    def __init__(self, model, shortlist_k: int = 8, request_limit: int = 16,
                 verbose: bool = False):
        self.model = model
        self.shortlist_k = shortlist_k
        self.request_limit = request_limit
        self.verbose = verbose
        self.decision_log: list[dict] = []

    def run(self, state: CampaignState, rng: np.random.Generator) -> None:
        agent = build_agent(self.model)
        deps = AgentDeps(state=state, shortlist_k=self.shortlist_k)
        memory, idle, round_no = "(none)", 0, 0
        while state.can_measure() and idle < 2:
            round_no += 1
            spent_before = state.ledger.spent
            log_from = len(state.ledger.log)
            prompt = ROUND_PROMPT.format(
                round=round_no, spent=state.ledger.spent, budget=state.ledger.budget,
                remaining=state.remaining, cost_lf=state.problem.cost_lf,
                cost_hf=state.problem.cost_hf, best=state.best_measured, memory=memory)
            # A failed round (API error, usage limit, retries exhausted) must not
            # kill the campaign: log it as a fallback round and let the idle guard
            # end things if the failures persist. Tools that already ran before the
            # failure have spent real budget, which the ledger reflects either way.
            try:
                result = agent.run_sync(
                    prompt, deps=deps,
                    usage_limits=UsageLimits(request_limit=self.request_limit))
                dec = result.output
                n_requests = result.usage.requests
                n_retries = sum(isinstance(p, RetryPromptPart)
                                for m in result.all_messages()
                                for p in getattr(m, "parts", []))
                error = None
            except Exception as exc:
                dec = RoundDecision(rationale=f"fallback: round failed ({exc})")
                n_requests = n_retries = None
                error = f"{type(exc).__name__}: {exc}"
            actions = state.ledger.log[log_from:]
            self.decision_log.append({
                "round": round_no, "spent_before": spent_before,
                "spent_after": state.ledger.spent, "budget": state.ledger.budget,
                "cost_lf": state.problem.cost_lf, "cost_hf": state.problem.cost_hf,
                "simulated": {o.candidate: o.value for o in actions if o.fidelity != HIGH},
                "measured": {o.candidate: o.value for o in actions if o.fidelity == HIGH},
                "n_requests": n_requests, "n_retries": n_retries,
                "fallback": error is not None, "error": error,
                "rationale": dec.rationale, "note": dec.note, "stop": dec.stop,
            })
            if self.verbose:
                print(f"    R{round_no} spent {state.ledger.spent - spent_before:g}: "
                      f"{dec.rationale}", flush=True)
            if dec.note:
                memory = (memory.removeprefix("(none)").strip() +
                          f"\nR{round_no}: {dec.note}").strip()
            if dec.stop:
                break
            # A round that spent nothing twice in a row means the agent is stuck
            # talking instead of experimenting; end the campaign rather than loop.
            idle = idle + 1 if state.ledger.spent == spent_before else 0
