"""Shared, dependency-light data types for the Elo tournament pipeline.

These primitives are imported by ``generate`` (which adds the GPU /
torch-heavy bits), ``match_builder``, ``run_tournament``, and the
tests. Keeping them in a torch-free module lets the orchestrator and
the unit tests run inside the default pixi env without the cuda stack.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Prompt:
    """One row of the prompt bank, normalised to a small typed shape."""

    prompt_id: str
    text: str
    source: str = ""
    subject: str | None = None
    gold_answer: str | None = None
    expected_behavior: str | None = None
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class GeneratedResponse:
    """Single (entrant, prompt) generation record persisted to the cache.

    Attributes:
        entrant_id: Stable id from :class:`EntrantSpec`.
        prompt_id: Stable id from the prompt bank.
        response_text: Decoded response text returned to the judge.
        p_correct: ``p_correct`` from the uncertainty head when the
            entrant requests the second-pass capture, otherwise
            ``None``. The configuration-level Elo run leaves this
            ``None`` for all entrants; the field is kept for forward
            compatibility with uncertainty-aware tournaments.
        diagnostics: Per-call metadata (PoE diagnostics or the
            HuggingFace mirror of the same schema).
        timestamp: ISO-8601 UTC timestamp of generation.
    """

    entrant_id: str
    prompt_id: str
    response_text: str
    p_correct: float | None
    diagnostics: dict[str, Any]
    timestamp: str


def prompt_seed(prompt_id: str) -> int:
    """Deterministic per-prompt seed used by the generation harness.

    A SHA-256 hash of the prompt id mod ``2**32`` — gives every prompt
    its own fixed seed so the random draft-head selection inside PoE
    lines up across the Hydra entrants on a given prompt while still
    varying across prompts.
    """
    return int(hashlib.sha256(prompt_id.encode("utf-8")).hexdigest(), 16) % (2**32)


def load_prompt_bank(path: Path) -> list[Prompt]:
    """Read a JSONL prompt bank into a list of :class:`Prompt`."""
    prompts: list[Prompt] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            prompts.append(
                Prompt(
                    prompt_id=entry["prompt_id"],
                    text=entry["text"],
                    source=entry.get("source", ""),
                    subject=entry.get("subject"),
                    gold_answer=entry.get("gold_answer"),
                    expected_behavior=entry.get("expected_behavior"),
                    tags=tuple(entry.get("tags", [])),
                )
            )
    return prompts


__all__ = [
    "GeneratedResponse",
    "Prompt",
    "load_prompt_bank",
    "prompt_seed",
]
