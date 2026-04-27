"""Unit tests for ``olmo_tap.final_evals.elo.match_builder``.

Pure-logic tests: no API calls, no model loading, no GPU. Cover pair
generation, swap-pair handling at the judge boundary, factuality
Source C filtering, missing-response skips, stratified pilot sampling
proportions, and reproducibility under a fixed seed.
"""

from __future__ import annotations

from collections import Counter
from typing import cast

import pytest

from olmo_tap.final_evals.elo.judge import CURATED_SOURCE, Dimension, PairToJudge
from olmo_tap.final_evals.elo.types import GeneratedResponse, Prompt
from olmo_tap.final_evals.elo.match_builder import (
    DEFAULT_PILOT_SEED,
    DEFAULT_PILOT_SIZE,
    PILOT_STRATA,
    build_match_list,
    select_pilot_subset,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _make_prompt(
    prompt_id: str,
    source: str,
    *,
    gold_answer: str | None = "Levothyroxine",
    expected_behavior: str | None = None,
    text: str | None = None,
) -> Prompt:
    return Prompt(
        prompt_id=prompt_id,
        text=text or f"Prompt {prompt_id}",
        source=source,
        subject="medicine",
        gold_answer=gold_answer,
        expected_behavior=expected_behavior,
        tags=(),
    )


def _make_response(
    entrant: str, prompt_id: str, text: str | None = None
) -> GeneratedResponse:
    return GeneratedResponse(
        entrant_id=entrant,
        prompt_id=prompt_id,
        response_text=text or f"{entrant} on {prompt_id}",
        p_correct=None,
        diagnostics={},
        timestamp="1970-01-01T00:00:00+00:00",
    )


def _full_cache(
    prompts: list[Prompt], entrants: list[str]
) -> dict[tuple[str, str], GeneratedResponse]:
    """Cache covering every (entrant, prompt) cell."""
    return {
        (e, p.prompt_id): _make_response(e, p.prompt_id)
        for p in prompts
        for e in entrants
    }


# --------------------------------------------------------------------------- #
# build_match_list — counts and pair generation
# --------------------------------------------------------------------------- #


def test_build_match_list_counts_per_dimension() -> None:
    """4 entrants × 3 prompts → 6 pairs × 3 prompts = 18 per dimension."""
    prompts = [
        _make_prompt("p1", "medmcqa_open"),
        _make_prompt("p2", "medqa"),
        _make_prompt("p3", "medqa"),
    ]
    entrants = ["alpha", "beta", "gamma", "delta"]
    cache = _full_cache(prompts, entrants)
    out = build_match_list(prompts, cache, entrants)

    assert set(out.keys()) == {"factuality", "calibration", "clinical_utility"}
    for dim in ("factuality", "calibration", "clinical_utility"):
        assert len(out[cast(Dimension, dim)]) == 18, (
            f"{dim}: expected 18 unordered pairs, got {len(out[cast(Dimension, dim)])}"
        )


def test_build_match_list_two_entrants_one_pair_per_prompt() -> None:
    prompts = [_make_prompt("p1", "medqa"), _make_prompt("p2", "medqa")]
    entrants = ["x", "y"]
    out = build_match_list(prompts, _full_cache(prompts, entrants), entrants)
    for pairs in out.values():
        assert len(pairs) == 2
        for pair in pairs:
            assert pair.entrant_a == "x"
            assert pair.entrant_b == "y"


def test_build_match_list_pair_ordering_follows_input_order() -> None:
    """itertools.combinations preserves the input ordering of entrants."""
    prompts = [_make_prompt("p1", "medqa")]
    entrants = ["delta", "alpha", "gamma"]  # deliberately unsorted
    out = build_match_list(prompts, _full_cache(prompts, entrants), entrants)
    pair_orderings = [(p.entrant_a, p.entrant_b) for p in out["calibration"]]
    assert pair_orderings == [
        ("delta", "alpha"),
        ("delta", "gamma"),
        ("alpha", "gamma"),
    ]


def test_build_match_list_populates_responses_from_cache() -> None:
    prompts = [_make_prompt("p1", "medqa", gold_answer="GOLD")]
    entrants = ["x", "y"]
    cache = {
        ("x", "p1"): _make_response("x", "p1", text="x-text"),
        ("y", "p1"): _make_response("y", "p1", text="y-text"),
    }
    out = build_match_list(prompts, cache, entrants)
    pair = out["factuality"][0]
    assert pair.response_a == "x-text"
    assert pair.response_b == "y-text"
    assert pair.gold_answer == "GOLD"
    assert pair.prompt_id == "p1"
    assert pair.source == "medqa"


def test_build_match_list_passes_through_expected_behavior() -> None:
    prompts = [
        _make_prompt(
            "p1", CURATED_SOURCE, gold_answer=None, expected_behavior="ABSTAIN"
        ),
    ]
    entrants = ["x", "y"]
    out = build_match_list(prompts, _full_cache(prompts, entrants), entrants)
    pair = out["calibration"][0]
    assert pair.expected_behavior == "ABSTAIN"
    assert pair.gold_answer is None


# --------------------------------------------------------------------------- #
# build_match_list — Source C filtering
# --------------------------------------------------------------------------- #


def test_factuality_drops_curated_prompts() -> None:
    prompts = [
        _make_prompt("medqa1", "medqa"),
        _make_prompt("curated1", CURATED_SOURCE, gold_answer=None),
        _make_prompt("medqa2", "medqa"),
    ]
    entrants = ["x", "y"]
    out = build_match_list(prompts, _full_cache(prompts, entrants), entrants)

    factuality_prompts = {p.prompt_id for p in out["factuality"]}
    assert factuality_prompts == {"medqa1", "medqa2"}

    for dim in ("calibration", "clinical_utility"):
        prompt_ids = {p.prompt_id for p in out[cast(Dimension, dim)]}
        assert prompt_ids == {"medqa1", "curated1", "medqa2"}


def test_factuality_filtering_with_only_curated_returns_empty() -> None:
    prompts = [_make_prompt("c1", CURATED_SOURCE, gold_answer=None)]
    entrants = ["x", "y"]
    out = build_match_list(prompts, _full_cache(prompts, entrants), entrants)
    assert out["factuality"] == []
    assert len(out["calibration"]) == 1
    assert len(out["clinical_utility"]) == 1


# --------------------------------------------------------------------------- #
# build_match_list — robustness against bad inputs
# --------------------------------------------------------------------------- #


def test_build_match_list_skips_prompts_missing_a_response(
    caplog: pytest.LogCaptureFixture,
) -> None:
    prompts = [
        _make_prompt("p1", "medqa"),
        _make_prompt("p2", "medqa"),
    ]
    entrants = ["x", "y", "z"]
    cache = _full_cache(prompts, entrants)
    # Drop one entrant's response for p1; p1 should be skipped entirely.
    cache.pop(("z", "p1"))

    with caplog.at_level("WARNING"):
        out = build_match_list(prompts, cache, entrants)

    for dim_pairs in out.values():
        prompt_ids = {p.prompt_id for p in dim_pairs}
        assert prompt_ids == {"p2"}
    assert any("p1" in rec.getMessage() for rec in caplog.records)


def test_build_match_list_rejects_too_few_entrants() -> None:
    prompts = [_make_prompt("p1", "medqa")]
    with pytest.raises(ValueError):
        build_match_list(prompts, {}, ["only_one"])


def test_build_match_list_rejects_duplicate_entrants() -> None:
    prompts = [_make_prompt("p1", "medqa")]
    with pytest.raises(ValueError):
        build_match_list(prompts, {}, ["x", "x"])


def test_build_match_list_dimensions_argument_filters_output() -> None:
    prompts = [_make_prompt("p1", "medqa")]
    entrants = ["x", "y"]
    out = build_match_list(
        prompts,
        _full_cache(prompts, entrants),
        entrants,
        dimensions=["factuality"],
    )
    assert set(out.keys()) == {"factuality"}
    assert len(out["factuality"]) == 1


# --------------------------------------------------------------------------- #
# select_pilot_subset — proportions and determinism
# --------------------------------------------------------------------------- #


def _build_full_bank() -> list[Prompt]:
    """Mirror the real bank's source mix: 24 + 69 + 50 = 143 prompts."""
    bank: list[Prompt] = []
    for i in range(24):
        bank.append(_make_prompt(f"srcA_{i:03d}", "medmcqa_open"))
    for i in range(69):
        bank.append(_make_prompt(f"srcB_{i:03d}", "medqa"))
    for i in range(50):
        bank.append(
            _make_prompt(
                f"srcC_{i:03d}",
                CURATED_SOURCE,
                gold_answer=None,
                expected_behavior="ABSTAIN",
            )
        )
    return bank


def test_pilot_subset_returns_50_with_correct_strata() -> None:
    bank = _build_full_bank()
    subset = select_pilot_subset(bank)
    assert len(subset) == DEFAULT_PILOT_SIZE
    counts = Counter(p.source for p in subset)
    assert counts == PILOT_STRATA


def test_pilot_subset_is_deterministic() -> None:
    bank = _build_full_bank()
    a = [p.prompt_id for p in select_pilot_subset(bank)]
    b = [p.prompt_id for p in select_pilot_subset(bank)]
    assert a == b


def test_pilot_subset_changes_with_seed() -> None:
    bank = _build_full_bank()
    a = {p.prompt_id for p in select_pilot_subset(bank, seed=DEFAULT_PILOT_SEED)}
    b = {p.prompt_id for p in select_pilot_subset(bank, seed=DEFAULT_PILOT_SEED + 1)}
    assert a != b
    # The strata are still respected even under a different seed.
    counts_a = Counter(
        p.source for p in select_pilot_subset(bank, seed=DEFAULT_PILOT_SEED + 1)
    )
    assert counts_a == PILOT_STRATA


def test_pilot_subset_preserves_bank_ordering() -> None:
    bank = _build_full_bank()
    subset = select_pilot_subset(bank)
    bank_order = {p.prompt_id: i for i, p in enumerate(bank)}
    indices = [bank_order[p.prompt_id] for p in subset]
    assert indices == sorted(indices)


def test_pilot_subset_rejects_unsupported_size() -> None:
    bank = _build_full_bank()
    with pytest.raises(ValueError):
        select_pilot_subset(bank, n=20)


def test_pilot_subset_raises_when_stratum_too_small() -> None:
    """Bank missing curated prompts should fail loudly, not silently."""
    bank = [_make_prompt(f"srcA_{i}", "medmcqa_open") for i in range(50)]
    with pytest.raises(ValueError):
        select_pilot_subset(bank)


# --------------------------------------------------------------------------- #
# Integration: pilot subset feeds build_match_list cleanly
# --------------------------------------------------------------------------- #


def test_pilot_subset_then_match_list_produces_expected_counts() -> None:
    bank = _build_full_bank()
    subset = select_pilot_subset(bank)
    entrants = ["a", "b", "c", "d"]
    cache = _full_cache(subset, entrants)
    out = build_match_list(subset, cache, entrants)

    n_curated = sum(1 for p in subset if p.source == CURATED_SOURCE)
    n_pairs = 6  # C(4, 2)

    assert len(out["factuality"]) == (DEFAULT_PILOT_SIZE - n_curated) * n_pairs
    assert len(out["calibration"]) == DEFAULT_PILOT_SIZE * n_pairs
    assert len(out["clinical_utility"]) == DEFAULT_PILOT_SIZE * n_pairs


def test_match_list_pairs_are_pairtojudge_instances() -> None:
    prompts = [_make_prompt("p1", "medqa")]
    entrants = ["x", "y"]
    out = build_match_list(prompts, _full_cache(prompts, entrants), entrants)
    for pairs in out.values():
        for pair in pairs:
            assert isinstance(pair, PairToJudge)
