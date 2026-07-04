"""A *genuinely agentic* BO policy: tools, memory, and multi-step deliberation.

The baseline "agentic" policy (``policies.AgenticBO``) is one stateless LLM call
per round that ranks ~8 pre-digested options — honestly *LLM-guided BO*. This
module implements LEARNINGS.md next-step 11: the model now **drives the round**
through a tool-calling loop, in two phases with (optionally) a different model
each — the interesting research question is whether a stronger *reasoner* at the
decision step, with a cheap *worker* doing the mechanical exploration, beats the
single-call agent.

Per round:

* **Investigate** (worker model) — a tool loop over the *real* BO state: the
  model queries the surrogate on any candidates it names (``predict``), searches
  the whole pool itself (``search_candidates``: by EI / mean / uncertainty /
  reagent match / random) instead of receiving a fixed shortlist, and reads its
  cross-round scratchpad (``recall``). It ends by calling ``report``.
* **Deliberate** (reasoner model) — given the investigation report, the measured
  history and its memory, it may ``predict`` a few candidates to verify a
  hypothesis, then ``submit`` the batch with a ``strategy``/``rationale`` and a
  ``note`` appended to memory for next round.

What is deliberately held fixed for a *fair* comparison (the §3 lesson again):
the batch size ``q`` and the total budget. The agent may *state* a preferred
``want_batch_size`` / ``want_stop`` — these are logged but not acted on in the
benchmark, so any measured win is attributable to the decision quality, not to
the agent quietly spending a different budget.

Backends: ``GeminiToolAgent`` (google-genai function calling, per-phase model
routing) and ``HeuristicToolAgent`` (deterministic, exercises the same tools and
logging offline — the knowledge-free floor and the CI path).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from .agent import _gemini_schema

# ---- tool JSON schemas (google-genai cleans out additionalProperties) --------

_SEARCH_TOOL = {
    "name": "search_candidates",
    "description": "Search the whole candidate pool yourself instead of using a fixed "
    "shortlist. Returns matching candidates (id + description).",
    "parameters": {
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "enum": ["top_ei", "top_mean", "most_uncertain", "random", "match"],
                "description": "top_ei/top_mean/most_uncertain rank the unmeasured pool by the "
                "surrogate; match filters descriptions by substring; random samples.",
            },
            "match": {
                "type": "string",
                "description": "Substring to match in candidate descriptions when mode=match, "
                "e.g. 'ligand=XPhos' or 'temp=90'.",
            },
            "limit": {"type": "integer", "description": "Max candidates to return (<=20)."},
        },
        "required": ["mode"],
    },
}

_PREDICT_TOOL = {
    "name": "predict",
    "description": "Query the surrogate model on candidates you name: returns posterior mean, "
    "std and expected improvement (EI) for each.",
    "parameters": {
        "type": "object",
        "properties": {
            "candidate_ids": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "Candidate ids to score (<=25).",
            }
        },
        "required": ["candidate_ids"],
    },
}

_RECALL_TOOL = {
    "name": "recall",
    "description": "Read your own scratchpad notes carried over from previous rounds.",
    "parameters": {"type": "object", "properties": {}},
}

_REPORT_TOOL = {
    "name": "report",
    "description": "Finish investigating. Summarise what you found and hand over a shortlist of "
    "the most promising candidate ids for the decision step.",
    "parameters": {
        "type": "object",
        "properties": {
            "findings": {"type": "string", "description": "Short summary of what you learned."},
            "shortlist": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "Candidate ids worth considering for this round's batch.",
            },
        },
        "required": ["findings", "shortlist"],
    },
}

_SUBMIT_TOOL = {
    "name": "submit",
    "description": "Commit this round's experiments and record a note for next round.",
    "parameters": {
        "type": "object",
        "properties": {
            "picks": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "Candidate ids to run this round, best first.",
            },
            "strategy": {
                "type": "string",
                "enum": ["explore", "exploit", "balance"],
            },
            "rationale": {"type": "string", "description": "One sentence explaining the choice."},
            "note": {
                "type": "string",
                "description": "A note to your future self (carried to later rounds), e.g. an "
                "observed pattern like 'solvent BuCN consistently underperforms'.",
            },
            "want_batch_size": {
                "type": "integer",
                "description": "If you could choose the batch size this round, what would it be? "
                "(Logged only; the benchmark keeps it fixed for fairness.)",
            },
            "want_stop": {
                "type": "boolean",
                "description": "Would you stop the campaign now if allowed? (Logged only.)",
            },
        },
        "required": ["picks", "strategy", "rationale", "note"],
    },
}

_WORKER_SYSTEM = (
    "You are the INVESTIGATION step of an autonomous experiment-design agent optimising a "
    "chemical reaction. Your job is to gather evidence, not to decide: use the tools to query "
    "the surrogate model on candidates you choose, search the candidate pool, and recall your "
    "notes. Be economical — a few targeted tool calls, then call report() with a shortlist for "
    "the decision step. Use any chemistry knowledge you have to steer the search."
)

_REASONER_SYSTEM = (
    "You are the DECISION step of an autonomous experiment-design agent optimising a chemical "
    "reaction, aiming to reach the highest yield in as few experiments as possible. You are "
    "given an investigation report, the measured history and your own notes. Optionally verify a "
    "hypothesis by calling predict() on a few candidates, then call submit() with your batch. "
    "Think about explore vs exploit given the remaining budget, and leave your future self a "
    "useful note. Use any domain knowledge you have."
)


@dataclass
class RoundDecision:
    """The outcome of one agentic round (returned by every tool-agent backend)."""

    picks: list                      # global candidate ids chosen this round (ordered)
    strategy: str | None = None
    rationale: str | None = None
    note: str = ""                   # appended to the campaign scratchpad
    want_batch_size: int | None = None
    want_stop: bool = False
    fallback: bool = False           # decision (partly) fell back to EI on error
    considered: list = field(default_factory=list)  # ids the agent predicted/searched
    tool_calls: list = field(default_factory=list)   # [{phase, name, args}] for auditing
    phase_models: dict = field(default_factory=dict)  # {"investigate": model, "decide": model}


class BOEnvironment:
    """Tool surface over one round's real BO state. All prices are pre-computed.

    The surrogate is fit once per round and scored on every unmeasured candidate,
    so ``predict``/``search_candidates`` are cheap dictionary look-ups no matter
    how the agent slices the pool.
    """

    def __init__(self, dataset, unseen, mean, std, ei, memory: str, rng):
        self._ds = dataset
        self._unseen = np.asarray(unseen)
        self._mean = {int(i): float(m) for i, m in zip(unseen, mean, strict=True)}
        self._std = {int(i): float(s) for i, s in zip(unseen, std, strict=True)}
        self._ei = {int(i): float(e) for i, e in zip(unseen, ei, strict=True)}
        self._memory = memory
        self._rng = rng
        self.calls: list = []            # tool-call log for the current phase
        self.considered: set = set()

    # -- tools ---------------------------------------------------------------
    def search_candidates(self, mode: str, match: str = "", limit: int = 10) -> list:
        limit = int(np.clip(limit, 1, 20))
        ids = self._unseen
        if mode == "match":
            want = (match or "").lower()
            ids = [i for i in ids if want in self._ds.descriptions[i].lower()]
            chosen = ids[:limit]
        elif mode == "random":
            k = min(limit, len(ids))
            chosen = [int(i) for i in self._rng.choice(ids, size=k, replace=False)]
        else:
            key = {"top_ei": self._ei, "top_mean": self._mean, "most_uncertain": self._std}[mode]
            chosen = sorted((int(i) for i in ids), key=lambda i: -key[i])[:limit]
        self.considered.update(chosen)
        return [{"id": int(i), "desc": self._ds.descriptions[i]} for i in chosen]

    def predict(self, candidate_ids: list) -> list:
        out = []
        for i in list(candidate_ids)[:25]:
            i = int(i)
            if i not in self._mean:
                continue  # already measured or out of range — silently skip
            self.considered.add(i)
            out.append({"id": i, "desc": self._ds.descriptions[i],
                        "mean": round(self._mean[i], 2), "std": round(self._std[i], 2),
                        "ei": round(self._ei[i], 3)})
        return out

    def recall(self) -> dict:
        return {"memory": self._memory or "(empty — this is an early round)"}

    # -- dispatch ------------------------------------------------------------
    def call(self, phase: str, name: str, args: dict):
        self.calls.append({"phase": phase, "name": name, "args": args})
        fn = {"search_candidates": self.search_candidates, "predict": self.predict,
              "recall": self.recall}[name]
        return fn(**args)

    def valid_id(self, i) -> bool:
        return int(i) in self._mean

    def ei_order(self) -> list:
        return sorted((int(i) for i in self._unseen), key=lambda i: -self._ei[i])


def _round_context(dataset, evaluated, y_obs, iteration, total_rounds, remaining_evals, q) -> str:
    """The measured-history briefing shared by both phases."""
    order = np.argsort(-np.asarray(y_obs))
    lines = [
        f"Round {iteration} of {total_rounds}. Experiments remaining after this one: "
        f"{remaining_evals - q}. You must choose {q} candidate(s) this round.",
        f"Objective to maximise: {dataset.objective_label}.",
        "",
        dataset.legend,
        "",
        "Measured so far (best first):",
    ]
    lines += [f"  id={evaluated[i]} {dataset.descriptions[evaluated[i]]}  ->  {y_obs[i]:.2f}"
              for i in order]
    lines.append(f"\nBest measured yield so far: {max(y_obs):.2f}.")
    return "\n".join(lines)


class HeuristicToolAgent:
    """Deterministic agentic policy: exercises every tool + the memory path, no API."""

    name = "heuristic-tools"

    def __init__(self):
        self.calls = 0
        self.fallbacks = 0
        self.cache_hits = 0
        self.decision_log: list = []

    def propose_round(self, env: BOEnvironment, ctx: str, q: int) -> RoundDecision:
        self.calls += 1
        env.recall()                                                  # touch memory
        env.call("investigate", "search_candidates", {"mode": "top_ei", "limit": 6})
        env.call("investigate", "search_candidates", {"mode": "most_uncertain", "limit": 2})
        scored = env.call("investigate", "predict",
                          {"candidate_ids": sorted(env.considered)})
        # UCB over what we looked at; explore early, exploit late.
        beta = max(0.1, 2.0 * (ctx.count("remaining") and 1.0))       # cheap, deterministic
        ranked = sorted(scored, key=lambda c: -(c["mean"] + beta * c["std"]))
        picks = [c["id"] for c in ranked[:q]]
        if len(picks) < q:
            for i in env.ei_order():
                if i not in picks:
                    picks.append(i)
                if len(picks) == q:
                    break
        best = ranked[0] if ranked else {"desc": "?", "mean": 0.0}
        return RoundDecision(
            picks=picks, strategy="balance",
            rationale="highest UCB among top-EI and most-uncertain candidates",
            note=f"UCB favourite so far: {best['desc']} (pred {best['mean']:.1f}).",
            considered=sorted(env.considered), tool_calls=list(env.calls),
            phase_models={"investigate": "heuristic", "decide": "heuristic"},
        )


class GeminiToolAgent:
    """Two-phase Gemini agent with per-phase model routing (google-genai)."""

    name = "gemini-tools"

    def __init__(self, worker_model: str = "gemini-2.5-flash",
                 reasoner_model: str = "gemini-2.5-flash", verbose: bool = False, cache=None,
                 max_worker_steps: int = 6, max_reasoner_steps: int = 4):
        import os

        from google import genai

        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.client = genai.Client(api_key=api_key)
        self.worker_model = worker_model
        self.reasoner_model = reasoner_model
        self.model = f"{worker_model}+{reasoner_model}"      # for cache keying / display
        self.verbose = verbose
        self.cache = cache
        self.max_worker_steps = max_worker_steps
        self.max_reasoner_steps = max_reasoner_steps
        self._warned = False
        self.calls = 0
        self.fallbacks = 0
        self.cache_hits = 0
        self.decision_log: list = []

    # -- one google-genai call, cached by (model, serialised transcript) ------
    def _call(self, model, system, tools, contents_repr, contents):
        from google.genai import types

        raw = None
        if self.cache is not None:
            key = self.cache.key("tools", model, system, json.dumps(tools, sort_keys=True),
                                 json.dumps(contents_repr, sort_keys=True))
            raw = self.cache.get(key)
        if raw is not None:
            self.cache_hits += 1
            return json.loads(raw)
        tool = types.Tool(function_declarations=[_gemini_schema(t) for t in tools])
        resp = self.client.models.generate_content(
            model=model, contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system, tools=[tool], temperature=0.0,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                tool_config=types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(mode="ANY")),
            ),
        )
        calls = [{"name": fc.name, "args": dict(fc.args or {})}
                 for fc in (resp.function_calls or [])]
        if self.cache is not None:
            self.cache.put(key, json.dumps(calls))
        return calls

    def _run_phase(self, model, system, tools, first_user, env, phase, terminal, max_steps):
        from google.genai import types

        contents = [types.Content(role="user", parts=[types.Part(text=first_user)])]
        contents_repr = [{"role": "user", "text": first_user}]
        for _ in range(max_steps):
            calls = self._call(model, system, tools, contents_repr, contents)
            if not calls:
                break
            model_parts = [types.Part(function_call=types.FunctionCall(name=c["name"],
                                                                       args=c["args"]))
                           for c in calls]
            contents.append(types.Content(role="model", parts=model_parts))
            contents_repr.append({"role": "model", "calls": calls})
            resp_parts, resp_repr = [], []
            for c in calls:
                if c["name"] == terminal:
                    return c["args"]
                result = env.call(phase, c["name"], c["args"])
                resp_parts.append(types.Part.from_function_response(
                    name=c["name"], response={"result": result}))
                resp_repr.append({"name": c["name"], "result": result})
            contents.append(types.Content(role="user", parts=resp_parts))
            contents_repr.append({"role": "tool", "responses": resp_repr})
        return None  # ran out of steps without a terminal call

    def propose_round(self, env: BOEnvironment, ctx: str, q: int) -> RoundDecision:
        self.calls += 1
        try:
            worker_user = (
                ctx + "\n\nInvestigate now: query the surrogate and search the pool with the "
                "tools, then call report() with a shortlist for the decision step."
            )
            report = self._run_phase(self.worker_model, _WORKER_SYSTEM,
                                     [_SEARCH_TOOL, _PREDICT_TOOL, _RECALL_TOOL, _REPORT_TOOL],
                                     worker_user, env, "investigate", "report",
                                     self.max_worker_steps) or {}

            # Give the decision step the investigation results as pre-scored context.
            shortlist = [int(i) for i in report.get("shortlist", []) if env.valid_id(i)]
            scored = env.predict(shortlist) if shortlist else env.predict(env.ei_order()[:8])
            mem = env.recall()["memory"]
            decide_user = (
                ctx
                + f"\n\nInvestigation findings: {report.get('findings', '(none)')}"
                + f"\n\nYour notes from earlier rounds:\n{mem}"
                + "\n\nShortlist with surrogate predictions (pred = mean +/- std, EI):\n"
                + "\n".join(f"  id={c['id']} {c['desc']}  ->  pred {c['mean']:.1f} "
                            f"+/- {c['std']:.1f}, EI {c['ei']:.3f}" for c in scored)
                + f"\n\nOptionally verify with predict(), then submit exactly {q} pick(s)."
            )
            submit = self._run_phase(self.reasoner_model, _REASONER_SYSTEM,
                                     [_PREDICT_TOOL, _SUBMIT_TOOL], decide_user, env,
                                     "decide", "submit", self.max_reasoner_steps)
            if submit is None:
                raise ValueError("no submit() from the reasoner")
            picks, fell_back = self._finalise_picks(submit.get("picks", []), env, q)
            if self.verbose:
                print(f"    gemini-tools[{submit.get('strategy', '?')}]: "
                      f"{submit.get('rationale', '')}")
            return RoundDecision(
                picks=picks, strategy=submit.get("strategy"),
                rationale=submit.get("rationale"), note=submit.get("note", ""),
                want_batch_size=submit.get("want_batch_size"),
                want_stop=bool(submit.get("want_stop", False)), fallback=fell_back,
                considered=sorted(env.considered), tool_calls=list(env.calls),
                phase_models={"investigate": self.worker_model, "decide": self.reasoner_model},
            )
        except Exception as exc:  # a transient API/format error must never crash a campaign
            if not self._warned:
                print(f"[warn] gemini-tools falling back to EI on error: {exc}")
                self._warned = True
            self.fallbacks += 1
            return RoundDecision(picks=env.ei_order()[:q], strategy="exploit",
                                 rationale="fallback: greedy EI", note="", fallback=True,
                                 considered=sorted(env.considered), tool_calls=list(env.calls),
                                 phase_models={"investigate": self.worker_model,
                                               "decide": self.reasoner_model})

    def _finalise_picks(self, raw_picks, env: BOEnvironment, q: int):
        picks, fell_back = [], False
        for i in raw_picks:
            if env.valid_id(i) and int(i) not in picks:
                picks.append(int(i))
            if len(picks) == q:
                break
        if len(picks) < q:  # top up from EI if the model under-picked / named bad ids
            fell_back = True
            self.fallbacks += 1
            for i in env.ei_order():
                if i not in picks:
                    picks.append(i)
                if len(picks) == q:
                    break
        return picks, fell_back
