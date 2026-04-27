from unittest.mock import patch

import pytest

pytest.importorskip("torch")

import torch

from app.backend.claim_confidence import compute_claim_confidences


def _fake_probs(rows: list[tuple[float, float, float]]) -> torch.Tensor:
    """Build a (N, 3) NLI probability tensor with columns [entailment, neutral, contradiction]."""
    return torch.tensor(rows, dtype=torch.float32)


def test_empty_claims_returns_empty_list():
    assert compute_claim_confidences("some response", [], object(), object()) == []


def test_entailment_maps_to_high_confidence():
    # One claim, fully entailed -> confidence ~1.0 -> "high"
    with patch(
        "kernel_entropy.nli.ModernBERTScorer.get_nli_probabilities",
        return_value=_fake_probs([(0.95, 0.04, 0.01)]),
    ):
        [metrics] = compute_claim_confidences(
            "response", ["claim"], bert_model=object(), bert_tokenizer=object()
        )
    assert metrics["level"] == "high"
    assert metrics["confidence"] == pytest.approx(0.97, abs=0.01)
    assert metrics["guidance"] == ""


def test_neutral_maps_to_low_confidence():
    # Neutral-heavy -> 0.5 * ~1.0 ~= 0.5 -> "low"
    with patch(
        "kernel_entropy.nli.ModernBERTScorer.get_nli_probabilities",
        return_value=_fake_probs([(0.10, 0.85, 0.05)]),
    ):
        [metrics] = compute_claim_confidences(
            "response", ["claim"], bert_model=object(), bert_tokenizer=object()
        )
    assert metrics["level"] == "low"
    assert metrics["confidence"] == pytest.approx(0.53, abs=0.01)


def test_contradiction_maps_to_low_confidence():
    with patch(
        "kernel_entropy.nli.ModernBERTScorer.get_nli_probabilities",
        return_value=_fake_probs([(0.05, 0.05, 0.90)]),
    ):
        [metrics] = compute_claim_confidences(
            "response", ["claim"], bert_model=object(), bert_tokenizer=object()
        )
    assert metrics["level"] == "low"
    assert metrics["confidence"] == pytest.approx(0.08, abs=0.01)


def test_preserves_order_across_claims():
    rows = [
        (0.90, 0.08, 0.02),  # 0.94 -> high
        (0.30, 0.60, 0.10),  # 0.60 -> low (just below moderate threshold of 0.65)
        (0.05, 0.10, 0.85),  # 0.10 -> low
    ]
    with patch(
        "kernel_entropy.nli.ModernBERTScorer.get_nli_probabilities",
        return_value=_fake_probs(rows),
    ):
        result = compute_claim_confidences(
            "response",
            ["a", "b", "c"],
            bert_model=object(),
            bert_tokenizer=object(),
        )

    assert [m["level"] for m in result] == ["high", "low", "low"]
    assert [m["confidence"] for m in result] == [
        pytest.approx(0.94, abs=0.01),
        pytest.approx(0.60, abs=0.01),
        pytest.approx(0.10, abs=0.01),
    ]


def test_scorer_receives_response_as_premise():
    """Each NLI pair must be (response, claim), not (claim, response)."""
    with patch(
        "kernel_entropy.nli.ModernBERTScorer.get_nli_probabilities",
        return_value=_fake_probs([(0.5, 0.5, 0.0), (0.5, 0.5, 0.0)]),
    ) as mock_nli:
        compute_claim_confidences(
            "full response text",
            ["claim one", "claim two"],
            bert_model=object(),
            bert_tokenizer=object(),
        )

    pairs = mock_nli.call_args[0][0]
    assert pairs == [
        ("full response text", "claim one"),
        ("full response text", "claim two"),
    ]
