"""A tiny persistent prompt->decision cache so interrupted LLM runs can resume.

The optimisation pipeline is deterministic (fixed seeds, temperature 0), so the
exact same prompt always maps to the same decision. Caching each decision by its
prompt lets a re-run of the same command reuse everything already computed and
only call the API for the steps that are genuinely new — turning a crashed or
Ctrl-C'd run into a free, instant resume.
"""
from __future__ import annotations

import hashlib
import json
import os


class PromptCache:
    def __init__(self, path: str):
        self.path = path
        self.data: dict = {}
        if path and os.path.exists(path):
            try:
                with open(path) as f:
                    self.data = json.load(f)
            except (json.JSONDecodeError, OSError):
                self.data = {}

    @staticmethod
    def key(*parts: str) -> str:
        return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()

    def get(self, key: str):
        return self.data.get(key)

    def put(self, key: str, value: str) -> None:
        self.data[key] = value
        self._save()

    def _save(self) -> None:
        if not self.path:
            return
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.data, f)
        os.replace(tmp, self.path)  # atomic: a crash mid-write never corrupts the cache
