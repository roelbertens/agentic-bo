"""Replay cache for LLM calls: rerunning an identical campaign is free.

Wraps any PydanticAI model. The key is a hash of the full serialized request
(instructions, tool schemas, message history), so any change to the prompt or
tools is a cache miss by construction — same contract as the other studies'
decision caches, one level lower in the stack.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import TypeAdapter
from pydantic_ai.messages import ModelMessagesTypeAdapter, ModelResponse
from pydantic_ai.models.wrapper import WrapperModel

_RESPONSE = TypeAdapter(ModelResponse)


# Fields that differ per run without changing what was asked: wall-clock stamps and
# the generated ids that tie tool calls, runs and conversations together.
_VOLATILE = {"timestamp", "tool_call_id", "run_id", "conversation_id", "id"}


def _canonical(node):
    if isinstance(node, dict):
        return {k: _canonical(v) for k, v in node.items() if k not in _VOLATILE}
    if isinstance(node, list):
        return [_canonical(v) for v in node]
    return node


class CachedModel(WrapperModel):
    """Persistent request -> response cache around a real model."""

    def __init__(self, wrapped, path: str | Path):
        super().__init__(wrapped)
        self.path = Path(path)
        self.hits = 0
        self.misses = 0
        self._store: dict[str, str] = {}
        if self.path.exists():
            self._store = json.loads(self.path.read_text())

    def _key(self, messages, model_request_parameters) -> str:
        raw = json.loads(ModelMessagesTypeAdapter.dump_json(messages))
        blob = (self.model_name
                + json.dumps(_canonical(raw), sort_keys=True)
                + repr(model_request_parameters))
        return hashlib.sha256(blob.encode()).hexdigest()

    async def request(self, messages, model_settings, model_request_parameters):
        key = self._key(messages, model_request_parameters)
        if key in self._store:
            self.hits += 1
            return _RESPONSE.validate_json(self._store[key])
        self.misses += 1
        response = await super().request(messages, model_settings, model_request_parameters)
        self._store[key] = _RESPONSE.dump_json(response).decode()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._store))
        return response
