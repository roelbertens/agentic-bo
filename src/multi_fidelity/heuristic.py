"""A deterministic stand-in for the LLM, as a PydanticAI ``FunctionModel``.

This is not a separate code path: the heuristic *is a model*, so every campaign
it drives goes through the real agent loop — instructions, tool schemas,
argument validation, retries, structured output — which is exactly what the
offline tests should exercise. It plays a sensible fixed strategy per round:

1. ``recall`` then ``shortlist``.
2. Judge the simulator from the LF/HF pairs seen so far: trusted while there is
   no evidence against it, distrusted once its mean absolute error exceeds half
   the spread of the measured values.
3. Trusted: simulate the top unsimulated shortlist candidates (keeping one
   measurement affordable), then measure the best simulated value. Distrusted:
   measure the top-EI candidate directly — degrade toward classic BO.
4. End the round; stop the campaign when a measurement is no longer affordable.
"""
from __future__ import annotations

import re

from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    RetryPromptPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

_SHORTLIST = re.compile(r"\[id=(\d+)\] EI=([-\d.]+) pred=([-\d.]+)\+/-([\d.]+) sim=(none|[-\d.]+)")
_BUDGET = re.compile(r"remaining=([-\d.]+) spent=[-\d.]+ cost_lf=([-\d.]+) cost_hf=([-\d.]+)")
_MEASURED = re.compile(r"\[id=(\d+)\] .* -> ([-\d.]+)")
_ERROR = re.compile(r"error=([+-][\d.]+)")
_UNCONFIRMED = re.compile(r"\[id=(\d+)\] sim=([-\d.]+)")
_RESULT = re.compile(r"\[id=(\d+)\] -> ([-\d.]+)")
_REMAINING = re.compile(r"remaining=([-\d.]+)")


def _round_returns(messages: list[ModelMessage]) -> tuple[dict[str, list[str]], bool]:
    """Tool returns of the current round (everything after the last user prompt),
    plus whether the round's last event was a bounced (retried) tool call."""
    returns: dict[str, list[str]] = {}
    retried = False
    for msg in messages:
        for part in getattr(msg, "parts", []):
            if isinstance(part, UserPromptPart):
                returns, retried = {}, False
            elif isinstance(part, ToolReturnPart):
                returns.setdefault(part.tool_name, []).append(str(part.content))
                retried = False
            elif isinstance(part, RetryPromptPart):
                retried = True
    return returns, retried


def _parse_recall(recall: str) -> dict:
    """Parse the sectioned ``recall`` text. The sections share the ``[id=..]``
    line shape, so matching must stay within each section — a pool-wide regex
    would, for example, read the simulator-vs-truth pairs (already measured) as
    unconfirmed candidates."""
    head, _, rest = recall.partition("Simulator vs truth")
    pairs_txt, _, unconfirmed_txt = rest.partition("Simulated only")
    return {
        "budget": tuple(map(float, _BUDGET.search(head).groups())),
        "measured_vals": [float(v) for _, v in _MEASURED.findall(head)],
        "errors": [abs(float(e)) for e in _ERROR.findall(pairs_txt)],
        "unconfirmed": [(int(c), float(v)) for c, v in _UNCONFIRMED.findall(unconfirmed_txt)],
    }


def _spread(values: list[float]) -> float:
    if len(values) < 2:
        return 1.0
    mean = sum(values) / len(values)
    return max((sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5, 1e-9)


def heuristic_model() -> FunctionModel:
    def call(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        done, retried = _round_returns(messages)
        final_tool = info.output_tools[0].name

        def finish(rationale: str, stop: bool) -> ModelResponse:
            return ModelResponse(parts=[ToolCallPart(
                final_tool, {"rationale": rationale, "note": "", "stop": stop})])

        # A bounced call would be recomputed identically (the heuristic is
        # deterministic), so give up on the round instead of burning retries.
        if retried:
            return finish("invalid call bounced; ending the round", stop=False)

        if "recall" not in done:
            return ModelResponse(parts=[ToolCallPart("recall", {})])
        if "shortlist" not in done:
            return ModelResponse(parts=[ToolCallPart("shortlist", {})])

        state = _parse_recall(done["recall"][-1])
        remaining, cost_lf, cost_hf = state["budget"]
        # The freshest budget figure: within a round, simulate precedes measure.
        for kind in ("simulate", "measure"):
            if kind in done:
                remaining = float(_REMAINING.search(done[kind][-1]).group(1))

        if "measure" in done:            # one measurement per round, then wrap up
            return finish("heuristic round done", stop=remaining < cost_hf)
        if remaining < cost_hf:
            return finish("cannot afford another measurement", stop=True)

        entries = [(int(c), float(ei), sim) for c, ei, _, _, sim
                   in _SHORTLIST.findall(done["shortlist"][-1])]
        errors = state["errors"]
        trusted = not errors or (sum(errors) / len(errors)) <= 0.5 * _spread(state["measured_vals"])

        if "simulate" in done:           # confirm the best of this round's screen
            sims = [(int(c), float(v)) for c, v in _RESULT.findall(done["simulate"][-1])]
            best = max(sims, key=lambda cv: cv[1])[0]
            return ModelResponse(parts=[ToolCallPart("measure", {"candidate_ids": [best]})])

        fresh = [c for c, _, sim in entries if sim == "none"]
        n_sim = min(3, len(fresh), int((remaining - cost_hf) // cost_lf))
        if trusted and n_sim >= 1:
            return ModelResponse(parts=[ToolCallPart(
                "simulate", {"candidate_ids": fresh[:n_sim]})])

        # Distrusted (or nothing left to screen): measure the most promising directly —
        # the best known simulated value if the simulator is trusted, else the top EI.
        simmed = state["unconfirmed"] + [(c, float(sim)) for c, _, sim in entries
                                         if sim != "none"]
        pick = max(simmed, key=lambda cv: cv[1])[0] if trusted and simmed else entries[0][0]
        return ModelResponse(parts=[ToolCallPart("measure", {"candidate_ids": [pick]})])

    return FunctionModel(call, model_name="heuristic")
