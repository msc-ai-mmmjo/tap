# app/backend/question_classifier.py
from typing import Literal

import torch
from transformers import AutoModelForSequenceClassification, PreTrainedTokenizerBase

QuestionType = Literal["mcq", "open", "none"]

_BERT_HYPOTHESES: dict[str, str] = {
    "mcq": "This is a multiple choice question",
    "open": "This is an open-ended question",
    "none": "This is not a question",
}


def classify_question_bert(
    model: AutoModelForSequenceClassification,
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    device: str = "cuda",
) -> QuestionType:
    entailment_idx = model.config.label2id.get("entailment", 2)
    scores: dict[str, float] = {}

    with torch.no_grad():
        for label, hypothesis in _BERT_HYPOTHESES.items():
            inputs = tokenizer(
                text,
                hypothesis,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).to(device)
            logits = model(**inputs).logits  # (1, num_labels)
            scores[label] = logits[0, entailment_idx].item()

    return max(scores, key=lambda k: scores[k])
