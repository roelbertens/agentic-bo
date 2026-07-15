# Test layers

Everything runs offline and deterministically where possible; randomness only
enters through seeded generators, and an LLM only enters at the outermost
layer. Each layer catches a class of problem the layers below cannot. The test
files live per study under `tests/`; the multi-fidelity study has the fullest
stack.

## 1. Unit tests with exact expected answers

Pure computation, no agent: given a concrete input, the test asserts the exact
output — budget arithmetic, curve extraction, oracle construction, parsers.
These pin the pieces everything else is built on.

## 2. The real agent loop, deterministically driven

The offline agent is not a separate code path: it is a deterministic strategy
packaged as a model (PydanticAI `FunctionModel`), so its tests exercise the
full production loop — instructions, tool schemas, argument validation,
retries, structured output — without an API. This layer checks the loop's
contracts: invalid tool calls are bounced back and charge nothing, a failing
model degrades to a logged fallback instead of crashing the campaign, a
replayed run through the request cache is free and identical, and every round
logs what it spent and asked.

## 3. Golden tests — pinned artifacts

Some artifacts are pinned exactly and may only change deliberately:

* **Prompts** — the caches are keyed on the exact request, so a casual
  rewording silently turns free replays into paid API calls.
* **Parsers** — regression tests hold parsing behaviour in place once a bug
  has been fixed.
* **One trajectory** — the exact action sequence of one short campaign.
  Property tests pass for any valid behaviour; a golden trajectory fails on
  *any* behaviour change (a logic edit, a tie-break shift, a library upgrade)
  and forces the change to be acknowledged in the diff.

## 4. Behavioural controls

Deterministic tests about behaviour rather than plumbing, built on knobs the
task design exposes. The central mechanism a study claims is pinned in
direction (the agent uses the simulator more when it is reliable), and the
knob's extremes double as controls: a setting where the input carries no
information (behaviour should collapse) and one where it is perfect.

## 5. Benchmark-level controls

The same discipline applied to results rather than code:

* **Baselines as controls** — a floor, a strong non-agentic ceiling, and the
  fixed strategies an adaptive method has to be located against.
* **Seeds with spread** — results are means over multiple seeds, so that noise
  is not read as a pattern.
* **Decision-log audits** — every agent round is logged (spend, picks, values,
  requests, retries, rationale), and audit scripts check the stated reasoning
  against the actual behaviour after the fact, with no API calls.

CI runs layers 1–4 plus lint on every push (`uv run pytest -q`); tests that
need the downloaded datasets skip when they are absent.
