# tests/test_question_classifier.py
from unittest.mock import MagicMock
import torch
from app.backend.question_classifier import classify_question_bert, classify_question_hydra


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


def _make_hydra_mocks(winning: str):
    """
    winning: one of 'MCQ', 'OPEN', 'NONE'
    Returns (model, tokenizer) mocks where the winning label token has the highest logit.
    """
    token_ids = {"MCQ": 100, "OPEN": 200, "NONE": 300}

    tokenizer = MagicMock()
    tokenizer.apply_chat_template.return_value = "fake prompt"
    tokenizer.encode.side_effect = lambda text, **kwargs: (
        [token_ids[text]] if text in token_ids else [1, 2, 3]
    )

    vocab_size = 400
    logits_tensor = torch.full((1, 1, 1, vocab_size), -10.0)
    logits_tensor[0, 0, 0, token_ids[winning]] = 10.0

    model = MagicMock()
    model.return_value = logits_tensor

    return model, tokenizer


def test_hydra_classifies_mcq():
    model, tokenizer = _make_hydra_mocks("MCQ")
    result = classify_question_hydra(model, tokenizer, "Which is correct? A. X B. Y", device="cpu")
    assert result == "mcq"


def test_hydra_classifies_open():
    model, tokenizer = _make_hydra_mocks("OPEN")
    result = classify_question_hydra(model, tokenizer, "What causes climate change?", device="cpu")
    assert result == "open"


def test_hydra_classifies_none():
    model, tokenizer = _make_hydra_mocks("NONE")
    result = classify_question_hydra(model, tokenizer, "Hello there.", device="cpu")
    assert result == "none"


def test_hydra_resets_kv_cache():
    model, tokenizer = _make_hydra_mocks("OPEN")
    classify_question_hydra(model, tokenizer, "any text", device="cpu")
    model.reset_kv_cache.assert_called_once()
