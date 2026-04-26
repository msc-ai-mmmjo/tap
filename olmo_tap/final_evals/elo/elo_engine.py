"""Permutation-averaged Elo with K-factor sensitivity sweep.

Implements the robustness recipe from Boubdir et al. (2023), *Elo Uncovered:
Robustness and Best Practices in Language Model Evaluation*:

  - Initial rating ``R_0 = 1400`` for every entrant.
  - Update rule
    ``R'_A = R_A + K * (S_A - E_A)`` with
    ``E_A = 1 / (1 + 10^((R_B - R_A) / 400))``.
  - Ties are dropped from the match list (consistent with the paper's
    handling of inconsistent judge verdicts).
  - The match list is shuffled ``n_perms`` times; per-permutation Elo is
    computed independently and we report mean ± SEM across permutations
    rather than a single-pass score.
  - A K-factor sweep returns the same per-entrant statistics over a list
    of K values, enabling the sensitivity heatmap (Figure 3 of the paper).

The input is a flat list of ``(entrant_a, entrant_b, winner)`` triples where
``winner`` is either one of the entrant ids or ``None`` / ``"TIE"`` to mark a
tie. See :func:`compute_elo_permutation` for the full contract.

Implementation notes:

  - All shuffling uses ``numpy.random.default_rng(seed)`` and the seed is
    surfaced on the result for reproducibility / logging.
  - The hot loop is written with simple Python and ``math.pow`` rather than
    NumPy because matches are processed serially per permutation; the
    speedup from vectorising would be marginal and would obscure the update
    rule.
  - Ratings are stored in a plain ``dict`` keyed by entrant id so the engine
    is agnostic to entrant ordering and to the entrant set.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

# Type alias for a single match record. ``winner`` is one of the two entrant
# ids on a decisive verdict, or ``None`` (canonical) / ``"TIE"`` for a tie.
Match = tuple[str, str, str | None]

DEFAULT_INITIAL_RATING: float = 1400.0
DEFAULT_K: float = 16.0
DEFAULT_N_PERMS: int = 500
DEFAULT_K_SWEEP: tuple[int, ...] = (1, 4, 8, 16, 32)
TIE_TOKEN: str = "TIE"


@dataclass(frozen=True)
class EloResult:
    """Per-entrant rating statistics across permutations.

    Attributes:
        entrant_id: The entrant the statistics are for.
        mean: Mean Elo rating across the permutations.
        sem: Standard error of the mean (``std(ddof=1) / sqrt(n_perms)``).
        ci95_low: ``mean - 1.96 * sem``.
        ci95_high: ``mean + 1.96 * sem``.
        per_perm_ratings: 1-D array of per-permutation final ratings; the
            full trace is returned so that downstream reports can plot
            distributions, not just summary statistics.
    """

    entrant_id: str
    mean: float
    sem: float
    ci95_low: float
    ci95_high: float
    per_perm_ratings: np.ndarray = field(repr=False)


def _normalise_match(match: Match) -> Match:
    """Coerce a tie marker into the canonical ``None`` form.

    Accepts ``None`` or the string ``"TIE"`` (case-insensitive) for the
    winner field; everything else must equal one of the two entrant ids and
    is returned unchanged.
    """
    a, b, winner = match
    if winner is None:
        return (a, b, None)
    if isinstance(winner, str) and winner.upper() == TIE_TOKEN:
        return (a, b, None)
    if winner != a and winner != b:
        raise ValueError(
            f"Match winner {winner!r} must be one of {a!r}, {b!r}, "
            f"None, or {TIE_TOKEN!r}."
        )
    return (a, b, winner)


def _drop_ties(matches: Iterable[Match]) -> list[Match]:
    """Filter out tied matches, returning a list of decisive matches."""
    out: list[Match] = []
    for m in matches:
        norm = _normalise_match(m)
        if norm[2] is None:
            continue
        out.append(norm)
    return out


def _expected_score(rating_a: float, rating_b: float) -> float:
    """Standard Elo expected-score for player A.

    ``E_A = 1 / (1 + 10 ** ((R_B - R_A) / 400))``.
    """
    return 1.0 / (1.0 + math.pow(10.0, (rating_b - rating_a) / 400.0))


def _entrant_ids(matches: Iterable[Match]) -> list[str]:
    """Return the unique entrant ids referenced by a match list, sorted."""
    seen: set[str] = set()
    for a, b, _ in matches:
        seen.add(a)
        seen.add(b)
    return sorted(seen)


def _run_single_pass(
    matches: Sequence[Match],
    entrants: Sequence[str],
    *,
    k: float,
    initial_rating: float,
) -> dict[str, float]:
    """Run a single sequential pass of the Elo update over ``matches``."""
    ratings: dict[str, float] = {e: initial_rating for e in entrants}
    for a, b, winner in matches:
        r_a = ratings[a]
        r_b = ratings[b]
        e_a = _expected_score(r_a, r_b)
        # winner is guaranteed to be either ``a`` or ``b`` after _drop_ties
        s_a = 1.0 if winner == a else 0.0
        s_b = 1.0 - s_a
        ratings[a] = r_a + k * (s_a - e_a)
        ratings[b] = r_b + k * (s_b - (1.0 - e_a))
    return ratings


def compute_elo_permutation(
    matches: Iterable[Match],
    *,
    k: float = DEFAULT_K,
    initial_rating: float = DEFAULT_INITIAL_RATING,
    n_perms: int = DEFAULT_N_PERMS,
    seed: int = 0,
) -> dict[str, EloResult]:
    """Compute permutation-averaged Elo ratings.

    The match list is shuffled ``n_perms`` times with
    :func:`numpy.random.default_rng(seed)`; for each permutation the Elo
    update is run sequentially and the final rating per entrant is recorded.
    The returned dict reports mean ± SEM and 95% CI per entrant alongside
    the full per-permutation trace.

    Args:
        matches: Iterable of ``(entrant_a, entrant_b, winner)`` triples.
            ``winner`` may be ``None`` or ``"TIE"`` to mark a tie; ties are
            dropped before the permutation loop.
        k: Elo K-factor.
        initial_rating: Starting rating for every entrant on every
            permutation. Boubdir et al. (2023) fix this at 1400.
        n_perms: Number of independent permutations to run. The paper
            recommends ``>= 100``; the headline run uses 500.
        seed: Seed for the NumPy ``Generator`` driving the shuffles. Logged
            on the result so runs are reproducible.

    Returns:
        Mapping ``entrant_id -> EloResult`` for every entrant referenced by
        ``matches``.

    Raises:
        ValueError: If ``n_perms < 1``, if ``matches`` is empty, or if any
            match has a winner that is neither one of the two entrants nor
            a recognised tie marker.
    """
    if n_perms < 1:
        raise ValueError(f"n_perms must be >= 1, got {n_perms}.")

    decisive = _drop_ties(matches)
    if not decisive:
        raise ValueError(
            "No decisive matches after dropping ties — "
            "cannot compute Elo over an empty match list."
        )

    entrants = _entrant_ids(decisive)
    rng = np.random.default_rng(seed)

    n = len(decisive)
    # Per-perm final ratings: row i = perm i, col j = entrants[j].
    final_ratings = np.empty((n_perms, len(entrants)), dtype=np.float64)

    for perm_idx in range(n_perms):
        order = rng.permutation(n)
        shuffled = [decisive[i] for i in order]
        ratings = _run_single_pass(
            shuffled, entrants, k=k, initial_rating=initial_rating
        )
        for col, entrant_id in enumerate(entrants):
            final_ratings[perm_idx, col] = ratings[entrant_id]

    results: dict[str, EloResult] = {}
    for col, entrant_id in enumerate(entrants):
        traces = final_ratings[:, col]
        mean = float(traces.mean())
        # ddof=1 for the unbiased estimator; SEM is undefined for n_perms == 1.
        if n_perms > 1:
            std = float(traces.std(ddof=1))
            sem = std / math.sqrt(n_perms)
        else:
            sem = 0.0
        results[entrant_id] = EloResult(
            entrant_id=entrant_id,
            mean=mean,
            sem=sem,
            ci95_low=mean - 1.96 * sem,
            ci95_high=mean + 1.96 * sem,
            per_perm_ratings=traces.copy(),
        )
    return results


def k_factor_sweep(
    matches: Iterable[Match],
    *,
    k_values: Sequence[int | float] = DEFAULT_K_SWEEP,
    initial_rating: float = DEFAULT_INITIAL_RATING,
    n_perms: int = DEFAULT_N_PERMS,
    seed: int = 0,
) -> dict[float, dict[str, EloResult]]:
    """Run :func:`compute_elo_permutation` across a list of K-factors.

    Returns a heatmap-ready ``{k: {entrant_id: EloResult}}`` structure. The
    same shuffle ``seed`` is used for every K so the difference across the
    sweep reflects the K-factor's effect rather than shuffle variance.

    Args:
        matches: Same contract as :func:`compute_elo_permutation`.
        k_values: K-factors to sweep. Default ``{1, 4, 8, 16, 32}`` covers
            the range where the Boubdir paper observed ranking changes on
            their benchmark suites.
        initial_rating: Starting rating per entrant per permutation.
        n_perms: Permutations per K-factor.
        seed: Seed for every K's permutation loop.

    Returns:
        Mapping ``k -> {entrant_id: EloResult}``. K-factors are stored as
        ``float`` keys for stable hashing across int / float inputs.
    """
    # Materialise once so each K sees the same input regardless of generator
    # exhaustion or list mutation.
    materialised = list(matches)
    sweep: dict[float, dict[str, EloResult]] = {}
    for k in k_values:
        sweep[float(k)] = compute_elo_permutation(
            materialised,
            k=float(k),
            initial_rating=initial_rating,
            n_perms=n_perms,
            seed=seed,
        )
    return sweep


def rank_entrants(
    results: dict[str, EloResult],
) -> list[tuple[str, float]]:
    """Return ``[(entrant_id, mean_rating), ...]`` sorted high-to-low."""
    return sorted(
        ((eid, r.mean) for eid, r in results.items()),
        key=lambda x: x[1],
        reverse=True,
    )
