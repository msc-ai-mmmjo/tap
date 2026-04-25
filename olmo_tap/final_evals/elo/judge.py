"""LLM-judge pipeline using the Anthropic Batch API with prompt caching.

Stubbed for now. The implementation will:

  - Load the rubric template for each dimension and split it into a
    cacheable system / rubric prefix and a per-pair user message.
  - Submit all judge queries via ``/v1/messages/batches`` rather than
    ``/v1/messages``. The 50% batch discount + ~10% prompt-cache rate are
    mandatory for the eval to fit a modest budget.
  - For every (pair, prompt, dimension) tuple, issue **two** queries with
    swapped position to mitigate position bias. A pair counts as a win
    only if the verdict is consistent across the swap; inconsistent
    verdicts collapse to ``TIE`` and are dropped from the Elo match list.
  - Hash on ``(entrant_a, entrant_b, prompt_id, dimension, position_order,
    rubric_version, judge_model)`` for the judgment cache key.

Type contracts are fixed here so :mod:`run_tournament` can be wired
against this module before the implementation lands.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

Verdict = Literal["A", "B", "TIE"]
Dimension = Literal["factuality", "calibration", "clinical_utility"]


@dataclass(frozen=True)
class JudgeQuery:
    """A single judge query (one position-order, one dimension).

    The full pair is judged twice (with positions swapped); each side is
    represented by its own :class:`JudgeQuery`.
    """

    prompt_id: str
    entrant_a: str
    entrant_b: str
    dimension: Dimension
    response_a: str
    response_b: str
    gold_answer: str | None
    position_swap: bool


@dataclass(frozen=True)
class JudgeVerdict:
    """A single judge response."""

    query: JudgeQuery
    verdict: Verdict
    reasoning: str
    cache_key: str
    judge_model: str


@dataclass(frozen=True)
class PairOutcome:
    """Aggregated outcome for one (pair, prompt, dimension) after position swap.

    The ``winner`` is one of the two entrant ids on a consistent verdict, or
    ``None`` for a tie / inconsistent verdict (which the Elo engine drops).
    """

    prompt_id: str
    dimension: Dimension
    entrant_a: str
    entrant_b: str
    winner: str | None
    raw_verdicts: tuple[JudgeVerdict, JudgeVerdict]


def build_judge_queries(
    pairs: list[tuple[str, str]],
    prompts: list[Mapping[str, Any]],
    responses: Mapping[tuple[str, str], str],
    dimensions: list[Dimension],
) -> list[JudgeQuery]:
    """Materialise the full grid of judge queries.

    Will iterate ``pairs × prompts × dimensions × {position_swap=False,
    position_swap=True}``.
    """
    raise NotImplementedError(
        "build_judge_queries is not yet implemented — pending the judge "
        "pipeline build-out."
    )


def submit_batch(
    queries: list[JudgeQuery],
    judge_model: str,
    rubric_paths: Mapping[Dimension, str],
) -> list[JudgeVerdict]:
    """Submit ``queries`` to the Anthropic Batch API and return verdicts.

    Will use the streaming batch poll loop documented in the Anthropic SDK
    and apply prompt caching on the system + rubric prefix.
    """
    raise NotImplementedError(
        "submit_batch is not yet implemented — pending the Anthropic Batch "
        "API integration."
    )


def collapse_position_swaps(
    verdicts: list[JudgeVerdict],
) -> list[PairOutcome]:
    """Reduce position-swapped verdict pairs to a single :class:`PairOutcome`.

    Inconsistent verdicts (A wins both orderings) collapse to ``winner=None``.
    """
    raise NotImplementedError(
        "collapse_position_swaps is not yet implemented — pending the judge "
        "pipeline build-out."
    )
