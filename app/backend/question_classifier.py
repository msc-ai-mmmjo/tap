from enum import StrEnum

import torch
from transformers import TokenizersBackend
from transformers.models.modernbert.modeling_modernbert import (
    ModernBertForSequenceClassification,
)


class QuestionType(StrEnum):
    MCQ = "mcq"
    OPEN = "open"


_BERT_HYPOTHESES: dict[QuestionType, str] = {
    QuestionType.MCQ: (
        "The question presents a small fixed set of answer choices to pick from, "
        "or can be answered with just 'yes' or 'no'."
    ),
    QuestionType.OPEN: (
        "The question expects a free-form explanatory answer and lists no options."
    ),
}


def detect_mcq_bert(
    model: ModernBertForSequenceClassification,
    tokenizer: TokenizersBackend,
    text: str,
    device: str = "cuda",
) -> bool:
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
