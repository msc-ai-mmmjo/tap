import math

import pytest

pytest.importorskip("torch")

import torch

from kernel_entropy.entropy import kle_from_similarity, kle_to_certainty


def test_kle_to_certainty_zero_entropy_is_full_certainty():
    assert kle_to_certainty(0.0, 5) == 1.0


def test_kle_to_certainty_max_entropy_is_zero_certainty():
    assert kle_to_certainty(math.log(5), 5) == pytest.approx(0.0, abs=1e-9)


def test_kle_to_certainty_midpoint():
    assert kle_to_certainty(math.log(5) / 2, 5) == pytest.approx(0.5, abs=1e-9)


def test_kle_to_certainty_requires_two_samples():
    assert kle_to_certainty(0.0, 1) is None


def test_kle_to_certainty_clamps_overflow_to_zero():
    # Slight numerical noise above log(N) must not produce a negative certainty.
    over = math.log(5) + 1e-6
    assert kle_to_certainty(over, 5) == 0.0


def test_kle_from_similarity_single_sample_is_zero():
    W = torch.zeros(1, 1)
    assert kle_from_similarity(W) == 0.0


def test_kle_from_similarity_zero_matrix_is_log_n():
    # W = 0 → L = 0 → K = I → ρ = I/n → eigenvalues all 1/n → VNE = log(n).
    n = 4
    W = torch.zeros(n, n)
    assert kle_from_similarity(W) == pytest.approx(math.log(n), abs=1e-6)


def test_kle_from_similarity_zero_lengthscale_is_log_n():
    # t=0 → K = exp(0) = I regardless of W → ρ = I/n → VNE = log(n).
    W = torch.tensor([[0.0, 0.7, 0.2], [0.7, 0.0, 0.5], [0.2, 0.5, 0.0]])
    assert kle_from_similarity(W, t=0.0) == pytest.approx(math.log(3), abs=1e-6)


def test_kle_from_similarity_is_non_negative():
    torch.manual_seed(0)
    for _ in range(3):
        n = 4
        A = torch.rand(n, n)
        W = (A + A.T) / 2  # symmetric, non-negative
        W.fill_diagonal_(0.0)
        assert kle_from_similarity(W) >= -1e-9


def test_kle_from_similarity_symmetric_invariance():
    W = torch.tensor([[0.0, 0.4, 0.1], [0.4, 0.0, 0.6], [0.1, 0.6, 0.0]])
    assert kle_from_similarity(W) == pytest.approx(kle_from_similarity(W.T), abs=1e-9)


def test_kle_from_similarity_bounded_above_by_log_n():
    # VNE of a unit-trace n×n density matrix is bounded above by log(n)
    # (achieved at maximally mixed state I/n).
    torch.manual_seed(7)
    n = 5
    A = torch.rand(n, n)
    W = (A + A.T) / 2
    W.fill_diagonal_(0.0)
    vne = kle_from_similarity(W)
    assert vne <= math.log(n) + 1e-6


def test_kle_from_similarity_two_samples_runs():
    # Smallest non-trivial case (n=2) should produce a finite, non-negative VNE.
    W = torch.tensor([[0.0, 0.5], [0.5, 0.0]])
    out = kle_from_similarity(W)
    assert math.isfinite(out)
    assert out >= -1e-9
