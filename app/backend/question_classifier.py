from enum import StrEnum

import torch
from transformers import (
    AutoModelForSequenceClassification,
    TokenizersBackend,
)


class QuestionType(StrEnum):
    MCQ = "mcq"
    OPEN = "open"


_BERT_HYPOTHESES: dict[QuestionType, str] = {
    QuestionType.MCQ: "This is a multiple choice question",
    QuestionType.OPEN: "This is not a multiple-choice question",
}


def detect_mcq_bert(
    model: AutoModelForSequenceClassification,
    tokenizer: TokenizersBackend,
    text: str,
    device: str = "cuda",
) -> bool:
    entailment_idx = model.config.label2id.get("entailment", 2)
    scores: dict[QuestionType, float] = {}

    with torch.no_grad():
        for label, hypothesis in _BERT_HYPOTHESES.items():
            inputs = tokenizer(
                text, hypothesis, return_tensors="pt", truncation=True, max_length=512
            ).to(device)
            logits = model(**inputs).logits  # (1, num_labels)
            scores[label] = logits[0, entailment_idx].item()

    return max(scores, key=lambda k: scores[k]) == QuestionType.MCQ
