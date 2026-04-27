"""Pairwise match list construction for the configuration-level Elo run.

Given the prompt bank, the per-entrant response cache, and the entrant
roster, this module produces the ``PairToJudge`` payloads consumed by
the LLM-judge pipeline. One ``PairToJudge`` is produced per
``(prompt, unordered entrant pair, dimension)`` triple; both position
orderings (A-vs-B and B-vs-A) are generated inside :func:`judge_pairs`
itself, so the builder is concerned with unique pairs only.

Two filtering rules are baked in:

  * Factuality requires a gold answer, so curated trustworthiness
    prompts (``source == "curated"``) are dropped from that dimension
    only. Calibration and clinical_utility evaluate framing and run on
    the full bank.
  * Prompts whose responses are missing for any participating entrant
    are dropped with a warning, so a partially-populated response cache
    still produces a coherent (smaller) match list rather than tripping
    a ``KeyError`` deep inside the judge call.

A second helper, :func:`select_pilot_subset`, draws a stratified
50-prompt subset for the pilot run.
"""

from __future__ import annotations

import logging
from itertools import combinations
from typing import Iterable

import numpy as np

from olmo_tap.final_evals.elo.judge import (
    CURATED_SOURCE,
    DIMENSIONS,
    Dimension,
    PairToJudge,
)
from olmo_tap.final_evals.elo.types import GeneratedResponse, Prompt

logger = logging.getLogger(__name__)


# Stratified pilot-subset shape — chosen to roughly preserve the source
# proportions of the full 143-prompt bank
# (24 medmcqa_open + 69 medqa + 50 curated → 8 + 24 + 18 = 50).
DEFAULT_PILOT_SIZE: int = 50
PILOT_STRATA: dict[str, int] = {
    "medmcqa_open": 8,
    "medqa": 24,
    "curated": 18,
}
DEFAULT_PILOT_SEED: int = 20260427


def _dedupe_entrants(entrants: list[str]) -> None:
    seen: set[str] = set()
    for eid in entrants:
        if eid in seen:
            raise ValueError(f"Duplicate entrant id in tournament: {eid!r}")
        seen.add(eid)


def _drop_factuality_curated(prompt: Prompt, dimension: Dimension) -> bool:
    """``True`` when the prompt should be excluded from this dimension's pairs."""
    return dimension == "factuality" and prompt.source == CURATED_SOURCE


def build_match_list(
    bank: list[Prompt],
    response_cache: dict[tuple[str, str], GeneratedResponse],
    entrants: list[str],
    dimensions: Iterable[Dimension] | None = None,
) -> dict[Dimension, list[PairToJudge]]:
    """Construct the per-dimension pair lists fed to the LLM judge.

    For each unordered pair of distinct entrants, each prompt in
    ``bank`` (after source filtering), and each dimension in
    ``dimensions``, one ``PairToJudge`` is produced. The canonical
    ``(entrant_a, entrant_b)`` ordering follows the input order of
    ``entrants``; the matching B-vs-A query is fired by
    :func:`judge_pairs` internally and is not represented here.

    Curated prompts (``source == "curated"``) are excluded from
    ``factuality`` because they have no gold answer; they are kept for
    ``calibration`` and ``clinical_utility``.

    Args:
        bank: Prompt bank loaded via :func:`generate.load_prompt_bank`.
        response_cache: Mapping ``(entrant_id, prompt_id) → response``.
            Prompts missing any entrant's response are skipped with a
            warning.
        entrants: Ordered list of entrant ids. Determines pair
            generation order via :func:`itertools.combinations`. Must
            contain at least two unique ids.
        dimensions: Dimensions to build pair lists for. Defaults to all
            three (``factuality``, ``calibration``, ``clinical_utility``).

    Returns:
        ``{dimension: [PairToJudge, ...]}``. Empty list for any
        dimension that has no surviving prompts.
    """
    if len(entrants) < 2:
        raise ValueError(
            f"Need at least two entrants to build a match list; got {entrants!r}."
        )
    _dedupe_entrants(entrants)

    dims: tuple[Dimension, ...] = (
        tuple(dimensions) if dimensions is not None else DIMENSIONS
    )
    out: dict[Dimension, list[PairToJudge]] = {d: [] for d in dims}

    for prompt in bank:
        responses: dict[str, GeneratedResponse] = {}
        missing: list[str] = []
        for eid in entrants:
            r = response_cache.get((eid, prompt.prompt_id))
            if r is None:
                missing.append(eid)
            else:
                responses[eid] = r
        if missing:
            logger.warning(
                "Skipping prompt %s: missing responses for %s",
                prompt.prompt_id,
                missing,
            )
            continue

        for dimension in dims:
            if _drop_factuality_curated(prompt, dimension):
                continue
            for entrant_a, entrant_b in combinations(entrants, 2):
                out[dimension].append(
                    PairToJudge(
                        prompt_id=prompt.prompt_id,
                        source=prompt.source,
                        prompt_text=prompt.text,
                        entrant_a=entrant_a,
                        entrant_b=entrant_b,
                        response_a=responses[entrant_a].response_text,
                        response_b=responses[entrant_b].response_text,
                        gold_answer=prompt.gold_answer,
                        expected_behavior=prompt.expected_behavior,
                    )
                )

    return out


def select_pilot_subset(
    bank: list[Prompt],
    n: int = DEFAULT_PILOT_SIZE,
    seed: int = DEFAULT_PILOT_SEED,
) -> list[Prompt]:
    """Draw a reproducible stratified subset of the prompt bank.

    The default 50-prompt sample preserves the source mix of the full
    143-prompt bank (24 medmcqa_open + 69 medqa + 50 curated) at
    ``PILOT_STRATA = {medmcqa_open: 8, medqa: 24, curated: 18}``. The
    output is deterministic given ``seed`` (default ``20260427``).

    Returned prompts follow the original bank ordering so downstream
    artifacts (matches.jsonl, pilot_summary.md) read consistently.

    Args:
        bank: Full prompt bank.
        n: Total subset size. Currently must equal
            :data:`DEFAULT_PILOT_SIZE`; adjusting the strata to a
            different ``n`` is a deliberate code change.
        seed: NumPy ``Generator`` seed for the per-stratum sample.

    Raises:
        ValueError: If ``n`` is not the supported pilot size, or if any
            stratum lacks the required number of prompts.
    """
    if n != DEFAULT_PILOT_SIZE:
        raise ValueError(
            f"select_pilot_subset is fixed at n={DEFAULT_PILOT_SIZE}; "
            f"got n={n}. Update PILOT_STRATA to use a different size."
        )

    by_source: dict[str, list[Prompt]] = {}
    for prompt in bank:
        by_source.setdefault(prompt.source, []).append(prompt)

    rng = np.random.default_rng(seed)
    chosen_ids: set[str] = set()
    for source, target_count in PILOT_STRATA.items():
        candidates = by_source.get(source, [])
        if len(candidates) < target_count:
            raise ValueError(
                f"Pilot stratum {source!r} requires {target_count} prompts; "
                f"only {len(candidates)} present in the bank."
            )
        idx = rng.choice(len(candidates), size=target_count, replace=False)
        for i in idx:
            chosen_ids.add(candidates[int(i)].prompt_id)

    return [p for p in bank if p.prompt_id in chosen_ids]


__all__ = [
    "DEFAULT_PILOT_SEED",
    "DEFAULT_PILOT_SIZE",
    "PILOT_STRATA",
    "build_match_list",
    "select_pilot_subset",
]
