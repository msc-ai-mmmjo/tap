"""Per-claim confidence via NLI self-entailment.

For every decomposed claim we compute P(response entails claim) using the
ModernBERT-large-nli scorer already loaded for Kernel Language Entropy. The
response is used as the premise and each claim as the hypothesis, so a high
score means the model's own output supports the claim.

The score is `P(entailment) + 0.5 * P(neutral)`, mirroring the KLE similarity
weighting. Because the three NLI probabilities sum to 1, this one-direction
score is bounded in [0, 1] (unlike the bidirectional KLE W matrix in [0, 2]).
We only use the response -> claim direction: the reverse direction is
uninformative here because a short claim generally cannot entail the full
response, so every score would collapse toward neutral/contradiction.

This is the degenerate single-sample case of SelfCheckGPT-NLI (Manakul et
al., EMNLP 2023): rather than sampling K responses and averaging entailment,
we treat the produced response as the sole reference context.
"""

from typing import Any


def score_to_metrics(score: float) -> dict:
    """Map a [0, 1] confidence score into the API's {confidence, level, guidance} dict."""
    score = round(float(score), 2)
    if score >= 0.80:
        return {"confidence": score, "level": "high", "guidance": ""}
    if score >= 0.65:
        return {
            "confidence": score,
            "level": "moderate",
            "guidance": "Verify with clinical reference",
        }
    return {
        "confidence": score,
        "level": "low",
        "guidance": "Cross-check with authoritative source before acting",
    }


def compute_claim_confidences(
    response: str,
    claims: list[str],
    bert_model: Any,
    bert_tokenizer: Any,
) -> list[dict]:
    """Score every claim against `response` with NLI.

    Returns one {confidence, level, guidance} dict per claim, in the same
    order as `claims`. Raises if the NLI forward pass fails; the caller is
    expected to treat the claim ledger as unavailable on exception.
    """
    if not claims:
        return []

    from kernel_entropy.nli import (
        LABEL_ENTAILMENT,
        LABEL_NEUTRAL,
        ModernBERTScorer,
    )

    scorer = ModernBERTScorer(
        sentences=claims,
        model=bert_model,
        tokenizer=bert_tokenizer,
    )
    pairs = [(response, claim) for claim in claims]
    probs = scorer.get_nli_probabilities(pairs)

    confidences = (probs[:, LABEL_ENTAILMENT] + 0.5 * probs[:, LABEL_NEUTRAL]).tolist()

    return [score_to_metrics(c) for c in confidences]
