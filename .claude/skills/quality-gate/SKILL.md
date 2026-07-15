---
name: quality-gate
description: Pre-commit quality gate for this public repo — run before committing or pushing any change. Verifies tests, lint, prompt/cache stability, docs sync, and public-repo hygiene (no secrets, no data files).
---

# Quality gate

Run every step below before a commit. Report the result as a short checklist
(pass/fail per step); a failed step blocks the commit until fixed.

## 1. Tests

```bash
uv run pytest -q
```

All tests must pass. Tests that need the reaction datasets auto-skip when
`data/` is absent (fetch with `uv run scripts/get_data.py`) — a *skip* is fine,
a *failure* never is.

## 2. Lint

```bash
uv run --group dev ruff check .
```

Zero findings. Fix code rather than adding ignores; per-file ignores in
`pyproject.toml` are reserved for the `sys.path` bootstrap pattern.

## 3. Prompt and cache stability

The decision cache (`.cache/`) is keyed by the **exact prompt text**. If the
diff touches `_build_prompt` in `src/agentic_bo/agent.py`, or the
`descriptions`/`legend` built in `src/datasets/reactions.py` or
`src/datasets/synthetic.py`:

- `tests/agentic_bo/test_credibility.py::test_default_prompt_format_is_stable` must be
  updated deliberately, never casually — changing the default prompt silently
  turns free cache resumes into paid API calls and breaks reproduction of the
  published numbers.
- Call this out explicitly in the commit message.

The same applies to the multi-fidelity study: its LLM replay cache is keyed on
the full request, so `INSTRUCTIONS`/`ROUND_PROMPT` in `src/multi_fidelity/agent.py`
are pinned by `tests/multi_fidelity/test_mf_agent.py::test_prompt_format_is_stable`.

## 4. Public-repo hygiene

- **No secrets:** scan the diff for API keys (`API_KEY=`, `sk-ant`, `AIza`,
  long base64-ish literals). Keys belong in the environment, never in code,
  logs, or committed results.
- **No data or cache:** `data/` and `.cache/` stay untracked (datasets are
  downloaded, not redistributed — that is a licensing statement in the README).
- **Results naming:** results live in a per-study folder
  (`results/agentic_bo/`, `results/rl_design/`, `results/multi_fidelity/`) and
  follow `<kind>_<dataset>[_persub][_anon][_permuted]_<agent>.<ext>` so runs
  never silently overwrite each other.

## 5. Docs stay in sync

- New or changed CLI flags in the runners (`run_agentic_bo.py`, `run_rl_design.py`,
  `run_multi_fidelity.py`) or `scripts/*.py` are reflected in `README.md` and the study docs.
- Changed benchmark numbers or conclusions are reflected in the study docs
  (`docs/AGENTIC_BO.md` / `docs/RL_DESIGN.md` / `docs/MULTI_FIDELITY.md`) — and only from
  clean runs (see the `experiment-hygiene` skill).
- The README `Layout` block matches the actual file tree.
