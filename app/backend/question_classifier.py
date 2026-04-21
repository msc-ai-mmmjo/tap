# app/backend/question_classifier.py
from typing import Literal

import torch

QuestionType = Literal["mcq", "open", "none"]

_BERT_HYPOTHESES: dict[str, str] = {
    "mcq": "This is a multiple choice question",
    "open": "This is an open-ended question",
    "none": "This is not a question",
}


def classify_question_bert(model, tokenizer, text: str, device: str = "cuda") -> QuestionType:
    entailment_idx = model.config.label2id.get("entailment", 2)
    scores: dict[str, float] = {}

    for label, hypothesis in _BERT_HYPOTHESES.items():
        inputs = tokenizer(
            text,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits  # (1, num_labels)
        scores[label] = logits[0, entailment_idx].item()

    return max(scores, key=lambda k: scores[k])
