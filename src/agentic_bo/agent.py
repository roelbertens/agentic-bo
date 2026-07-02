"""The decision-making core of the *agentic* policy.

Three interchangeable backends implement the same ``select_batch`` interface:

* ``HeuristicAgent`` — deterministic, dependency-free. Runs anywhere (CI,
  offline). Has no domain knowledge, so it is the floor for the agentic loop.
* ``GeminiAgent``    — Google Gemini via the ``google-genai`` SDK.
* ``AnthropicAgent`` — Claude via the Anthropic SDK.

Each receives the same ``DecisionContext`` and returns a ranked list of
*positions* into the shortlist (each 0..k-1). The agentic policy in
``policies.py`` maps those back to global candidate indices. The context is
dataset-agnostic: candidates and history
are passed as human-readable descriptions, so the same prompt works for the
synthetic MOF task and the real reaction dataset.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

# JSON schema for the structured decision (used by both LLM backends).
# The agent returns a ranked batch of shortlist indices ("picks", best first),
# supporting batch optimisation (q candidates per round); q=1 is a length-1 list.
DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "picks": {
            "type": "array",
            "items": {"type": "integer"},
            "description": "Shortlist indices (0-based) to run next, best first. "
            "Provide exactly the requested number of distinct candidates.",
        },
        "strategy": {
            "type": "string",
            "enum": ["explore", "exploit", "balance"],
            "description": "Whether this round prioritises reducing uncertainty, chasing the "
            "highest predicted objective, or a balance of both.",
        },
        "rationale": {
            "type": "string",
            "description": "One short sentence explaining the choice.",
        },
    },
    "required": ["picks", "strategy", "rationale"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = (
    "You are an optimisation policy for autonomous scientific discovery. Each round you "
    "choose which candidate to run/measure next, aiming to find the candidate with the "
    "highest objective value in as few experiments as possible. Use any domain knowledge "
    "you have about the candidates, and balance exploration (reducing uncertainty in "
    "under-sampled regions) against exploitation (high predicted objective). Respond only "
    "via the provided structured schema."
)


@dataclass
class DecisionContext:
    """Everything a policy needs to choose the next experiment this round."""

    iteration: int
    budget: int
    remaining: int
    objective_label: str
    legend: str
    hist_desc: list            # human-readable descriptions of measured candidates
    y_eval: np.ndarray         # (n_eval,) measured objective values (aligned to hist_desc)
    X_eval: np.ndarray         # (n_eval, d) features already measured (heuristic use)
    cand_desc: list            # human-readable descriptions of the shortlist
    cand_ids: list             # global dataset indices of the shortlist
    X_cand: np.ndarray         # (k, d) features of the shortlist (heuristic use)
    mean: np.ndarray | None    # (k,) GP posterior mean, or None (ablation)
    std: np.ndarray | None     # (k,) GP posterior std, or None (ablation)
    ei: np.ndarray | None      # (k,) expected improvement, or None (ablation)
    use_surrogate: bool
    n_select: int = 1          # how many candidates to pick this round (batch size)

    @property
    def best_value(self) -> float:
        return float(self.y_eval.max())


def _fallback_order(ctx: DecisionContext) -> list:
    """Ranking to fall back on: greedy EI, else greedy mean, else shortlist order."""
    if ctx.ei is not None:
        return list(np.argsort(-ctx.ei))
    if ctx.mean is not None:
        return list(np.argsort(-ctx.mean))
    return list(range(len(ctx.cand_ids)))


def _take(order, n, k):
    """First n distinct positions from `order`, each in [0, k)."""
    out = []
    for p in order:
        p = int(p)
        if 0 <= p < k and p not in out:
            out.append(p)
        if len(out) == n:
            break
    return out


class HeuristicAgent:
    """Deterministic stand-in that mimics an agentic explore/exploit policy."""

    name = "heuristic"

    def select_batch(self, ctx: DecisionContext) -> list:
        if ctx.use_surrogate and ctx.mean is not None and ctx.std is not None:
            # UCB-style: weight uncertainty more early on, exploit as the budget runs out.
            beta = max(0.1, 2.0 * ctx.remaining / max(1, ctx.budget))
            order = np.argsort(-(ctx.mean + beta * ctx.std))
        else:
            # No surrogate: rank by closeness to the best candidate seen so far.
            x_best = ctx.X_eval[int(np.argmax(ctx.y_eval))]
            order = np.argsort(np.linalg.norm(ctx.X_cand - x_best, axis=1))
        return _take(order, ctx.n_select, len(ctx.cand_ids))


class _LLMAgent:
    """Shared plumbing for LLM backends: prompt, cache, parse, validate, fall back."""

    name = "llm"
    model = ""

    def __init__(self, verbose: bool = False, cache=None):
        self.verbose = verbose
        self.cache = cache
        self._warned = False
        self.calls = 0         # total decisions requested
        self.fallbacks = 0     # decisions that fell back (API/parse error)
        self.cache_hits = 0    # decisions served from the persistent cache

    def _raw_decision(self, prompt: str) -> str:  # pragma: no cover - overridden
        raise NotImplementedError

    def select_batch(self, ctx: DecisionContext) -> list:
        self.calls += 1
        prompt = _build_prompt(ctx)
        picks = []
        try:
            raw, hit = None, False
            if self.cache is not None:
                key = self.cache.key(self.name, self.model, prompt)
                raw = self.cache.get(key)
            if raw is not None:
                hit = True
                self.cache_hits += 1
            else:
                raw = self._raw_decision(prompt)
                if self.cache is not None:
                    self.cache.put(key, raw)
            data = json.loads(raw)
            picks = _take(data.get("picks", []), ctx.n_select, len(ctx.cand_ids))
            if picks and self.verbose and not hit:  # keep resumes (all cache hits) quiet
                print(f"    {self.name}[{data.get('strategy', '?')}]: "
                      f"{data.get('rationale', '')}")
        except Exception as exc:  # transient API / parse errors must never crash a run
            if not self._warned:
                print(f"[warn] {self.name} agent falling back to EI/heuristic on error: {exc}")
                self._warned = True
        if len(picks) < ctx.n_select:  # top up from the fallback ranking if under-filled
            self.fallbacks += 1
            for p in _fallback_order(ctx):
                if p not in picks:
                    picks.append(p)
                if len(picks) == ctx.n_select:
                    break
        return picks


def _gemini_schema(schema):
    """Return a copy of a JSON schema that Gemini accepts.

    Gemini's structured-output parser rejects ``additionalProperties`` (Anthropic
    requires it), so strip it at every level.
    """
    if isinstance(schema, dict):
        return {k: _gemini_schema(v) for k, v in schema.items()
                if k not in ("additionalProperties", "additional_properties")}
    if isinstance(schema, list):
        return [_gemini_schema(v) for v in schema]
    return schema


class GeminiAgent(_LLMAgent):
    """Google Gemini as the optimisation policy, via the google-genai SDK."""

    name = "gemini"

    def __init__(self, model: str = "gemini-2.5-flash", verbose: bool = False, cache=None):
        super().__init__(verbose, cache)
        import os

        from google import genai  # lazy import; only this backend needs it

        self._genai = genai
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
        self.model = model

    def _raw_decision(self, prompt: str) -> str:
        from google.genai import types

        resp = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=_gemini_schema(DECISION_SCHEMA),
                temperature=0.0,
            ),
        )
        return resp.text


class AnthropicAgent(_LLMAgent):
    """Claude as the optimisation policy, via the Anthropic Messages API."""

    name = "claude"

    def __init__(self, model: str = "claude-opus-4-8", effort: str = "low",
                 verbose: bool = False, cache=None):
        super().__init__(verbose, cache)
        import anthropic  # lazy import; only this backend needs it

        self.client = anthropic.Anthropic()
        self.model = model
        self.effort = effort

    def _raw_decision(self, prompt: str) -> str:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=512,
            system=SYSTEM_PROMPT,
            output_config={
                "effort": self.effort,
                "format": {"type": "json_schema", "schema": DECISION_SCHEMA},
            },
            messages=[{"role": "user", "content": prompt}],
        )
        return next(b.text for b in resp.content if b.type == "text")


def _build_prompt(ctx: DecisionContext) -> str:
    """Render the run history and shortlist as a compact, human-readable prompt."""
    lines: list[str] = []
    lines.append(
        f"Experiment {ctx.iteration} of {ctx.budget}. "
        f"Evaluations remaining after this one: {ctx.remaining - 1}."
    )
    lines.append(f"Objective to maximise: {ctx.objective_label}.")
    if ctx.legend:
        lines.append("")
        lines.append(ctx.legend)
    lines.append("")

    lines.append("Candidates measured so far (best first):")
    order = np.argsort(-ctx.y_eval)
    for i in order:
        lines.append(f"  {ctx.hist_desc[i]}  ->  {ctx.y_eval[i]:.2f}")
    lines.append("")
    lines.append(f"Current best measured value: {ctx.best_value:.2f}.")
    lines.append("")

    if ctx.use_surrogate:
        lines.append(
            "Candidate shortlist with surrogate-model predictions "
            "(pred = posterior mean +/- std, EI = expected improvement):"
        )
    else:
        lines.append("Candidate shortlist (no surrogate model available — use your own knowledge):")
    for pos in range(len(ctx.cand_ids)):
        extra = ""
        if ctx.use_surrogate and ctx.mean is not None:
            extra = f"  ->  pred {ctx.mean[pos]:.2f} +/- {ctx.std[pos]:.2f}, EI {ctx.ei[pos]:.3f}"
        lines.append(f"  [{pos}] {ctx.cand_desc[pos]}{extra}")
    lines.append("")
    if ctx.n_select == 1:
        lines.append(
            "Choose exactly one candidate by its [index], returned as a one-element 'picks' "
            "list. Prefer the pick that makes the most progress toward the global optimum "
            "given the remaining budget."
        )
    else:
        lines.append(
            f"Choose the {ctx.n_select} best distinct candidates to run together this round, "
            f"by their [index], returned as a 'picks' list ordered best first. Pick a set that "
            f"makes the most progress toward the global optimum given the remaining budget."
        )
    return "\n".join(lines)
