"""Unit tests for ``olmo_tap.final_evals.elo.elo_engine``.

Covers four properties of the permutation-averaged Elo engine:

  1. Two-player Bernoulli — true ``p=0.7`` over 1000 games puts the
     stronger player ~150 Elo above the weaker player with small SEM.
  2. Three-player transitivity — A>B and B>C at 70% each implies A>C
     under permutation-averaged Elo.
  3. Permutation stability — different shuffles produce different
     single-pass ratings but similar means once we average over many
     permutations.
  4. K-factor sensitivity — for a clear-winner Bernoulli the ranking is
     stable at K in {1, 4, 8, 16} and noise amplification widens the
     per-permutation rating spread at K=32.
"""

from __future__ import annotations

import random

from olmo_tap.final_evals.elo.elo_engine import (
    DEFAULT_INITIAL_RATING,
    EloResult,
    Match,
    _run_single_pass,
    compute_elo_permutation,
    k_factor_sweep,
    rank_entrants,
)


def _bernoulli_matches(
    a: str,
    b: str,
    p_a_wins: float,
    n: int,
    seed: int,
) -> list[Match]:
    """Generate ``n`` Bernoulli matches between ``a`` and ``b``."""
    rng = random.Random(seed)
    matches: list[Match] = []
    for _ in range(n):
        winner = a if rng.random() < p_a_wins else b
        matches.append((a, b, winner))
    return matches


def test_two_player_bernoulli_separates_by_about_150_elo() -> None:
    matches = _bernoulli_matches("A", "B", p_a_wins=0.7, n=1000, seed=1)
    results = compute_elo_permutation(matches, k=16, n_perms=100, seed=42)

    assert set(results) == {"A", "B"}
    a, b = results["A"], results["B"]

    # Mean A > Mean B by a substantial margin (theoretical equilibrium for
    # p=0.7 is about 147 Elo).  Loose bounds make the test robust to seeds.
    delta = a.mean - b.mean
    assert 100.0 < delta < 220.0, f"unexpected delta: {delta}"

    # SEM should be small — rating variance across permutations is bounded
    # because the true win rate is fixed.
    assert a.sem < 5.0
    assert b.sem < 5.0

    # Ratings should be roughly symmetric around the initial rating; total
    # mass is conserved by the symmetric update rule.
    midpoint = (a.mean + b.mean) / 2
    assert abs(midpoint - DEFAULT_INITIAL_RATING) < 1e-6

    # 95% CIs are sane (low < high, high - low ≈ 3.92 * SEM).
    assert a.ci95_low < a.ci95_high
    assert abs((a.ci95_high - a.ci95_low) - 3.92 * a.sem) < 1e-6


def test_three_player_transitivity_under_permutations() -> None:
    matches: list[Match] = []
    matches.extend(_bernoulli_matches("A", "B", p_a_wins=0.7, n=500, seed=2))
    matches.extend(_bernoulli_matches("B", "C", p_a_wins=0.7, n=500, seed=3))
    # Add some direct A vs C matches so the engine can pick up the gap
    # without relying on a single chained pass.
    matches.extend(_bernoulli_matches("A", "C", p_a_wins=0.8, n=500, seed=4))

    results = compute_elo_permutation(matches, k=16, n_perms=200, seed=7)

    ranking = [eid for eid, _ in rank_entrants(results)]
    assert ranking == ["A", "B", "C"], ranking

    # A should clear C by a wide margin even after permutation averaging.
    assert results["A"].mean - results["C"].mean > 150.0


def test_permutation_stability_means_match_across_seeds() -> None:
    matches = _bernoulli_matches("A", "B", p_a_wins=0.7, n=600, seed=11)

    # Two single-pass Elos with hand-picked orderings — should differ.
    forward = _run_single_pass(
        matches, ["A", "B"], k=16, initial_rating=DEFAULT_INITIAL_RATING
    )
    backward = _run_single_pass(
        list(reversed(matches)),
        ["A", "B"],
        k=16,
        initial_rating=DEFAULT_INITIAL_RATING,
    )
    # Different ordering of the same Bernoulli draws gives non-identical
    # ratings — this order-dependence is exactly what permutation averaging
    # exists to smooth out.
    assert forward["A"] != backward["A"]

    # Two permutation-averaged runs with different seeds should agree
    # closely on the mean per-entrant rating.
    res_seed0 = compute_elo_permutation(matches, k=16, n_perms=200, seed=0)
    res_seed1 = compute_elo_permutation(matches, k=16, n_perms=200, seed=99)

    for entrant_id in ("A", "B"):
        m0 = res_seed0[entrant_id].mean
        m1 = res_seed1[entrant_id].mean
        # Tolerance generous enough to hold across reasonable seeds and tight
        # enough to catch a regression in the averaging logic.
        assert abs(m0 - m1) < 6.0, (entrant_id, m0, m1)


def test_k_factor_sensitivity_stable_low_k_unstable_high_k() -> None:
    matches = _bernoulli_matches("A", "B", p_a_wins=0.7, n=1000, seed=5)
    sweep = k_factor_sweep(matches, k_values=(1, 4, 8, 16, 32), n_perms=100, seed=13)

    # Ranking is stable at low K factors.
    for k in (1.0, 4.0, 8.0, 16.0):
        ranking = [eid for eid, _ in rank_entrants(sweep[k])]
        assert ranking == ["A", "B"], (k, ranking)

    # For a fixed match set, larger K amplifies per-permutation rating
    # variance (each game moves ratings further), so the spread of
    # single-permutation final ratings widens with K.  We verify this
    # monotonically here as a more direct probe of noise amplification than
    # ranking flips on a clean 0.7-Bernoulli (which stays stable everywhere).
    def _spread(per_perm: EloResult) -> float:
        traces = per_perm.per_perm_ratings
        return float(traces.max() - traces.min())

    spread_low = _spread(sweep[1.0]["A"])
    spread_high = _spread(sweep[32.0]["A"])
    assert spread_high > 2.0 * spread_low, (spread_low, spread_high)


def test_ties_dropped_from_match_list() -> None:
    matches: list[Match] = [
        ("A", "B", "A"),
        ("A", "B", None),
        ("A", "B", "TIE"),
        ("A", "B", "B"),
    ]
    results = compute_elo_permutation(matches, k=16, n_perms=10, seed=0)
    # Two decisive matches — A and B each won once — final mean per entrant
    # collapses back to the initial rating.
    assert abs(results["A"].mean - DEFAULT_INITIAL_RATING) < 1.0
    assert abs(results["B"].mean - DEFAULT_INITIAL_RATING) < 1.0


def test_compute_elo_permutation_validates_inputs() -> None:
    import pytest

    with pytest.raises(ValueError):
        compute_elo_permutation([("A", "B", "A")], n_perms=0)

    with pytest.raises(ValueError):
        # All matches dropped as ties.
        compute_elo_permutation([("A", "B", None)], n_perms=10)

    with pytest.raises(ValueError):
        # Winner doesn't reference either entrant.
        compute_elo_permutation([("A", "B", "C")], n_perms=10)
