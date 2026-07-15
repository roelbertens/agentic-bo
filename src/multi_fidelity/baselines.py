"""The non-agentic policies: what the agent has to beat.

Ordered by how much they know about the second oracle:

* ``RandomHF``   — ignore the simulator, measure at random. The floor.
* ``ClassicBOHF``— ignore the simulator, run plain GP + EI on measurements only.
  The strongest single-fidelity baseline; any multi-fidelity method that loses
  to this wasted its simulation budget.
* ``TwoStage``   — the human heuristic: spend a fixed fraction of the budget
  screening random candidates in simulation, then measure the screen's top
  scorers. Uses the simulator maximally but *trusts it blindly* — no model of
  how wrong it is, and no way back if it is junk.
* ``MFBO``       — multi-fidelity BO: one GP over (features, fidelity flag) fit
  on both kinds of observation, EI at the measurement fidelity, and a
  cheap-first escalation rule — simulate the EI-argmax while the model is still
  uncertain about its simulated value, measure it once it is not. Adapts to the
  simulator's quality through the kernel, at zero LLM cost.

Every policy stops once it can no longer afford a measurement: only measured
values count as results, so cheap spend that can never be confirmed is waste.
"""
from __future__ import annotations

import numpy as np

from .campaign import CampaignState
from .surrogate import GaussianSurrogate, MultiFidelitySurrogate, expected_improvement


class RandomHF:
    name = "random_hf"

    def run(self, state: CampaignState, rng: np.random.Generator) -> None:
        while state.can_measure() and len(um := state.unmeasured()):
            state.measure([int(rng.choice(um))])


class ClassicBOHF:
    name = "classic_bo_hf"

    def run(self, state: CampaignState, rng: np.random.Generator) -> None:
        X = state.problem.dataset.X
        while state.can_measure() and len(um := state.unmeasured()):
            ids = np.fromiter(state.hf_obs, dtype=int)
            y = np.array([state.hf_obs[int(i)] for i in ids])
            mean, std = GaussianSurrogate(seed=0).fit(X[ids], y).predict(X[um])
            ei = expected_improvement(mean, std, best=float(y.max()))
            state.measure([int(um[int(np.argmax(ei))])])


class TwoStage:
    """Screen-then-confirm with a fixed split of the budget."""

    name = "two_stage"

    def __init__(self, screen_frac: float = 0.5):
        self.screen_frac = screen_frac

    def run(self, state: CampaignState, rng: np.random.Generator) -> None:
        screen_budget = self.screen_frac * state.ledger.budget
        while (state.ledger.spent < screen_budget and state.can_simulate()
               and len(us := state.unsimulated())):
            state.simulate([int(rng.choice(us))])
        # Confirm the screen's ranking top-down with the remaining budget.
        ranked = sorted(state.lf_obs, key=state.lf_obs.__getitem__, reverse=True)
        for c in ranked:
            if not state.can_measure():
                break
            if c not in state.hf_obs:
                state.measure([c])
        # Leftover budget with the screen exhausted: fall back to random measuring.
        while state.can_measure() and len(um := state.unmeasured()):
            state.measure([int(rng.choice(um))])


class MFBO:
    """GP + EI with a learned fidelity relation and cheap-first escalation.

    Each round: fit the augmented GP on everything, take the EI-argmax at the
    measurement fidelity, then choose *how* to query it. If its simulated value
    is still uncertain (LF posterior std above ``gamma`` times the spread of the
    measurements) and a simulation is affordable, simulate — the GP interpolates
    the result, so the same candidate escalates to a measurement on a later
    round. Otherwise measure. With an uninformative simulator the kernel
    decorrelates the fidelities, LF stops reducing LF-std of unseen points, and
    the rule degrades toward classic BO instead of chasing a junk landscape.
    """

    name = "mf_bo"

    def __init__(self, gamma: float = 0.5):
        self.gamma = gamma

    def run(self, state: CampaignState, rng: np.random.Generator) -> None:
        while state.can_measure() and len(um := state.unmeasured()):
            surr = MultiFidelitySurrogate(state.problem.dataset.X, seed=0)
            surr.fit(state.lf_obs, state.hf_obs)
            mean, std = surr.predict_hf(um)
            ei = expected_improvement(mean, std, best=state.best_measured)
            pick = int(um[int(np.argmax(ei))])
            hf_spread = float(np.std(list(state.hf_obs.values()))) or 1.0
            lf_std = float(surr.predict_lf(np.array([pick]))[1][0])
            if (pick not in state.lf_obs and state.can_simulate()
                    and lf_std > self.gamma * hf_spread):
                state.simulate([pick])
            else:
                state.measure([pick])
