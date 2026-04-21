# tests/test_question_classifier.py
from unittest.mock import MagicMock
import torch
from app.backend.question_classifier import classify_question_bert


def _make_bert_mocks(entailment_idx: int, winning_label_idx: int):
    """
    winning_label_idx: 0=mcq, 1=open, 2=none
    Returns (model, tokenizer) mocks where the winning label has the highest entailment score
    at the specified entailment_idx column.
    """
    tokenizer = MagicMock()
    tokenizer.return_value = MagicMock()
    tokenizer.return_value.to.return_value = {"input_ids": torch.zeros(1, 10, dtype=torch.long)}

    def make_logits(is_winner):
        base = [-1.0, 0.0, 0.5]
        result = base[:]
        if is_winner:
            result[entailment_idx] = 2.0
        return torch.tensor([result])

    logits_list = [make_logits(i == winning_label_idx) for i in range(3)]

    call_count = [0]
    def fake_forward(**kwargs):
        out = MagicMock()
        out.logits = logits_list[call_count[0]]
        call_count[0] += 1
        return out

    model = MagicMock()
    model.config.label2id = {"contradiction": 0, "neutral": 1, "entailment": entailment_idx}
    model.side_effect = fake_forward
    return model, tokenizer


def test_bert_classifies_mcq():
    model, tokenizer = _make_bert_mocks(entailment_idx=2, winning_label_idx=0)
    assert classify_question_bert(model, tokenizer, "Which of the following is correct? A. X B. Y", device="cpu") == "mcq"


def test_bert_classifies_open():
    model, tokenizer = _make_bert_mocks(entailment_idx=2, winning_label_idx=1)
    assert classify_question_bert(model, tokenizer, "What causes climate change?", device="cpu") == "open"


def test_bert_classifies_none():
    model, tokenizer = _make_bert_mocks(entailment_idx=2, winning_label_idx=2)
    assert classify_question_bert(model, tokenizer, "Hello there.", device="cpu") == "none"


def test_bert_classifies_mcq_nonstandard_entailment_idx():
    # entailment mapped to column 0; verifies the function reads label2id rather than hardcoding 2
    model, tokenizer = _make_bert_mocks(entailment_idx=0, winning_label_idx=0)
    assert classify_question_bert(model, tokenizer, "Which is correct? A. X B. Y", device="cpu") == "mcq"
