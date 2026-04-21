# app/backend/question_classifier.py
from typing import Literal

import torch
from transformers import AutoModelForSequenceClassification, PreTrainedTokenizerBase

from olmo_tap.hydra import HydraTransformer

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


_CLASSIFICATION_SYSTEM_PROMPT = (
    "Classify the user's input. Respond with exactly one word: "
    "'MCQ' if it contains a multiple choice question with labelled options, "
    "'OPEN' if it is an open-ended question, "
    "or 'NONE' if it is not a question."
)

_HYDRA_LABELS: dict[str, QuestionType] = {
    "MCQ": "mcq",
    "OPEN": "open",
    "NONE": "none",
}


def classify_question_hydra(
    model: HydraTransformer,
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    device: str = "cuda",
) -> QuestionType:
    messages = [
        {"role": "system", "content": _CLASSIFICATION_SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]
    chat_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = torch.tensor([tokenizer.encode(chat_prompt)], device=device)

    model.reset_kv_cache()
    with torch.no_grad():
        logits = model(input_ids, return_logits=True)  # (n_heads, batch, seq, vocab)

    first_token_logits = logits[0, 0, -1, :]  # head 0, batch 0, last position

    candidate_token_ids = {
        label: tokenizer.encode(label_str, add_special_tokens=False)[0]
        for label_str, label in _HYDRA_LABELS.items()
    }
    scores = {label: first_token_logits[tid].item() for label, tid in candidate_token_ids.items()}
    return max(scores, key=lambda k: scores[k])
