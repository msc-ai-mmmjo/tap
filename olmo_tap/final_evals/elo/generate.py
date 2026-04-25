"""Entrant -> response generation for the Elo tournament.

Stubbed for now. The implementation will:

  - Load each entrant via :func:`olmo_tap.final_evals.elo.entrants.build_entrant`.
  - For ``vanilla_hf`` entrants: call HuggingFace ``model.generate(..., do_sample=False)``.
  - For ``custom_poe`` entrants: call
    ``poe.generate_with_cache(prompt_text=..., compute_uncertainty=...,
    seed=prompt_seed(prompt_id), bypass_jury=spec.bypass_jury,
    temperature=spec.temperature)`` and snapshot ``poe.last_diagnostics``.
  - Persist responses + diagnostics under ``caches/responses/`` keyed on a
    content hash of the entrant config + prompt text.
  - Free GPU memory between loader swaps via ``del model; gc.collect();
    torch.cuda.empty_cache()`` per the existing eval-script convention.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping

from olmo_tap.final_evals.elo.entrants import EntrantSpec


@dataclass(frozen=True)
class GeneratedResponse:
    """Single (entrant, prompt) generation record persisted to the cache.

    Attributes:
        entrant_id: Stable id from :class:`EntrantSpec`.
        prompt_id: Stable id from the prompt bank.
        response: The decoded response text returned to the judge.
        diagnostics: Free-form mapping populated from
            ``PoE.last_diagnostics`` plus any caller-side metadata
            (selected draft head, seed, n_resampled, n_tokens, etc.).
        p_correct: Optional ``p_correct`` score from the uncertainty head;
            None for entrants that don't request the second-pass capture.
    """

    entrant_id: str
    prompt_id: str
    response: str
    diagnostics: Mapping[str, Any]
    p_correct: float | None


def prompt_seed(prompt_id: str) -> int:
    """Deterministic per-prompt seed used by :func:`generate_entrant_response`.

    A SHA-256 hash of the prompt id mod 2**32 — gives every prompt its own
    fixed seed so the random draft-head selection inside PoE lines up
    across the Hydra entrants on a given prompt while still varying across
    prompts. Re-running the eval reproduces the same canonical response per
    ``(entrant, prompt)``.
    """
    return int(hashlib.sha256(prompt_id.encode("utf-8")).hexdigest(), 16) % (2**32)


def generate_entrant_response(
    entrant: EntrantSpec,
    prompt: Mapping[str, Any],
    model_handle: Any,
) -> GeneratedResponse:
    """Generate a single response for ``(entrant, prompt)``.

    ``model_handle`` is whatever :func:`build_entrant` returns once wired up
    — likely a tuple of ``(model, tokenizer)`` or a wrapped :class:`PoE`.
    """
    raise NotImplementedError(
        "generate_entrant_response is not yet wired up — depends on "
        "build_entrant and the eval-mode PoE kwargs."
    )


def generate_for_tournament(
    entrants: list[EntrantSpec],
    prompts: list[Mapping[str, Any]],
    cache_dir: str,
) -> list[GeneratedResponse]:
    """Drive generation across the full ``entrants × prompts`` grid.

    The implementation will iterate entrant-by-entrant (so a single model is
    loaded at a time), share the loaded model between the two robustness-
    LoRA entrants, and write each record to ``cache_dir`` immediately so
    partial runs are resumable.
    """
    raise NotImplementedError(
        "generate_for_tournament is not yet wired up — depends on the "
        "build_entrant dispatch and the response cache layout."
    )
