"""
Zero-shot MCQ vs open-ended classifier built on the ModernBERT-NLI model.

The same NLI checkpoint loaded by :func:`app.backend.bert_inference.load_bert`
is reused here as a zero-shot text classifier: each candidate class is
encoded as a hypothesis and the class with the highest entailment logit
wins. The result drives the system-prompt and token-budget routing inside
:func:`app.backend.hydra_inference.generate`.
"""

from enum import StrEnum

import torch
from transformers import TokenizersBackend
from transformers.models.modernbert.modeling_modernbert import (
    ModernBertForSequenceClassification,
)


class QuestionType(StrEnum):
    """Possible classification outcomes for a user prompt."""

    MCQ = "mcq"
    OPEN = "open"


_BERT_HYPOTHESES: dict[QuestionType, str] = {
    QuestionType.MCQ: "This is a multiple choice question",
    QuestionType.OPEN: "This is not a multiple-choice question",
}


def detect_mcq_bert(
    model: ModernBertForSequenceClassification,
    tokenizer: TokenizersBackend,
    text: str,
    device: str = "cuda",
) -> bool:
    """
    Zero-shot MCQ classification by entailment scoring.

    For each :class:`QuestionType` we score ``(text, hypothesis)`` through
    the NLI head and read off the entailment logit. Returns ``True`` iff
    the MCQ hypothesis scores higher than the open-ended one.

    :param model: ModernBERT-NLI model from
        :func:`app.backend.bert_inference.load_bert`.
    :param tokenizer: Matching tokenizer.
    :param text: User prompt.
    :param device: Torch device for the forward pass.

    :returns: ``True`` if the prompt looks like a multiple-choice question.
    """
    # should be {'contradiction': 2, 'entailment': 0, 'neutral': 1}
    if (label_id_map := model.config.label2id) is not None:
        entailment_idx = label_id_map.get("entailment", 0)
    else:
        entailment_idx = 0

    scores: dict[QuestionType, float] = {}

    with torch.no_grad():
        for label, hypothesis in _BERT_HYPOTHESES.items():
            inputs = tokenizer(
                text, hypothesis, return_tensors="pt", truncation=True, max_length=512
            ).to(device)
            logits = model(**inputs).logits  # (1, num_labels)
            scores[label] = logits[0, entailment_idx].item()

    return max(scores, key=lambda k: scores[k]) == QuestionType.MCQ
