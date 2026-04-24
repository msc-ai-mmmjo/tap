from unittest.mock import MagicMock, patch

import pytest
from transformers import TokenizersBackend

from app.backend.hydra_inference import (
    MCQ_SYSTEM_PROMPT,
    NLP_SYSTEM_PROMPT,
    _tokens_and_resamples_from_poe_output,
    generate,
    load_hydra,
)
from olmo_tap.constants import MAX_NEW_TOKENS, MCQ_MAX_NEW_TOKENS


def test_generate_emits_tokens_and_resamples():
    mock_model = MagicMock()
    mock_model.heads = [MagicMock()] * 10  # 9 LLM + 1 uncertainty
    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template.return_value = "<prompt>"
    mock_tokenizer.eos_token_id = 7
    mock_tokenizer.decode.return_value = "<eos>"

    output_parts = ["<chat-prefix>", "Hello", " world", "<eos>"]
    original_tokens = ["universe"]
    resampled_idxs = [2]

    with patch("app.backend.hydra_inference.PoE") as MockPoE:
        MockPoE.return_value.generate_with_cache.return_value = (
            output_parts,
            original_tokens,
            resampled_idxs,
            None,  # NLP: uncertainty is None
        )
        raw, tokens, resampled, uncertainty = generate(
            mock_model,
            mock_tokenizer,
            [{"role": "user", "content": "say hi"}],
            is_mcq=False,
            device="cpu",
        )

    assert raw == "Hello world"
    assert tokens == ["Hello", "world"]
    assert resampled == [
        {
            "index": 1,
            "old_token": "universe",
            "new_token": "world",
            "severity": 1.0,
        }
    ]
    assert uncertainty is None

    ctor_kwargs = MockPoE.call_args.kwargs
    assert ctor_kwargs["n_llm_heads"] == 9
    assert ctor_kwargs["max_new_tokens"] == MAX_NEW_TOKENS

    call_kwargs = MockPoE.return_value.generate_with_cache.call_args.kwargs
    assert call_kwargs["is_mcq"] is False
    assert call_kwargs["messages"] == [
        {"role": "system", "content": NLP_SYSTEM_PROMPT},
        {"role": "user", "content": "say hi"},
    ]


def test_generate_mcq_injects_system_prompt_and_short_budget():
    mock_model = MagicMock()
    mock_model.heads = [MagicMock()] * 10
    mock_tokenizer = MagicMock()
    mock_tokenizer.eos_token_id = 7
    mock_tokenizer.decode.return_value = "<eos>"

    output_parts = ["<chat-prefix>", "A"]

    with patch("app.backend.hydra_inference.PoE") as MockPoE:
        MockPoE.return_value.generate_with_cache.return_value = (
            output_parts,
            [],
            [],
            0.83,  # MCQ: uncertainty is a float p_correct
        )
        raw, tokens, _, uncertainty = generate(
            mock_model,
            mock_tokenizer,
            [{"role": "user", "content": "Which fits? A) Gout B) ..."}],
            is_mcq=True,
            device="cpu",
        )

    assert raw == "A"
    assert tokens == ["A"]
    assert uncertainty == 0.83

    ctor_kwargs = MockPoE.call_args.kwargs
    assert ctor_kwargs["n_llm_heads"] == 9
    assert ctor_kwargs["max_new_tokens"] == MCQ_MAX_NEW_TOKENS

    call_kwargs = MockPoE.return_value.generate_with_cache.call_args.kwargs
    assert call_kwargs["is_mcq"] is True
    assert call_kwargs["messages"][0] == {
        "role": "system",
        "content": MCQ_SYSTEM_PROMPT,
    }
    assert call_kwargs["messages"][1]["role"] == "user"


def test_tokens_and_resamples_no_resamples():
    mock_tokenizer = MagicMock()
    mock_tokenizer.eos_token_id = 7
    mock_tokenizer.decode.return_value = "<eos>"

    output_parts = ["<prefix>", "Hello", " world"]
    raw, tokens, resampled = _tokens_and_resamples_from_poe_output(
        mock_tokenizer, output_parts, [], []
    )

    assert raw == "Hello world"
    assert tokens == ["Hello", "world"]
    assert resampled == []


def test_tokens_and_resamples_strips_trailing_eos():
    mock_tokenizer = MagicMock()
    mock_tokenizer.eos_token_id = 7
    mock_tokenizer.decode.return_value = "<eos>"

    output_parts = ["<prefix>", "Hi", "<eos>"]
    raw, tokens, resampled = _tokens_and_resamples_from_poe_output(
        mock_tokenizer, output_parts, [], []
    )

    assert raw == "Hi"
    assert tokens == ["Hi"]
    assert resampled == []


def test_tokens_and_resamples_drops_eos_resample():
    mock_tokenizer = MagicMock()
    mock_tokenizer.eos_token_id = 7
    mock_tokenizer.decode.return_value = "<eos>"

    output_parts = ["<prefix>", "Hi", "<eos>"]
    original_tokens = ["draft_eos"]
    resampled_idxs = [2]

    raw, tokens, resampled = _tokens_and_resamples_from_poe_output(
        mock_tokenizer, output_parts, original_tokens, resampled_idxs
    )

    assert raw == "Hi"
    assert tokens == ["Hi"]
    assert resampled == []


def test_load_hydra_returns_model_and_tokenizer():
    mock_model = MagicMock()
    mock_tokenizer = MagicMock(spec=TokenizersBackend)

    with (
        patch("app.backend.hydra_inference.WEIGHTS_DIR", "fake_weights"),
        patch(
            "app.backend.hydra_inference.load_ensemble",
            return_value=(mock_model, 10),
        ),
        patch("app.backend.hydra_inference.AutoTokenizer") as mock_auto,
    ):
        mock_auto.from_pretrained.return_value = mock_tokenizer
        result = load_hydra(device="cpu")

    assert len(result) == 2
    assert result[0] is mock_model
    assert result[1] is mock_tokenizer


def test_get_robustness_nlp_worst_case(monkeypatch):
    """NLP path: worst_case is the suffix with the lowest NLI similarity."""
    import torch

    from app.backend.hydra_inference import get_robustness

    suffix_bank = ["suffixA", "suffixB", "suffixC"]
    adv_responses = {
        "suffixA": "response A",
        "suffixB": "response B",  # lowest score -> worst case
        "suffixC": "response C",
    }
    scripted_scores = {
        "response A": 1.8,
        "response B": 0.3,
        "response C": 1.2,
    }

    def fake_generate(model, tokenizer, messages, is_mcq, device):
        last = messages[-1]["content"]
        for suffix, resp in adv_responses.items():
            if last.endswith(suffix):
                return (resp, [], [], None)
        raise AssertionError(f"unexpected adv prompt: {last}")

    class FakeScorer:
        def __init__(self, sentences, model, tokenizer):
            self._sentences = sentences

        def compute_against_baseline(self, baseline_idx=0):
            n = len(self._sentences)
            result = torch.zeros(n)
            for j in range(n):
                if j != baseline_idx:
                    result[j] = scripted_scores[self._sentences[j]]
            return result

    monkeypatch.setattr("app.backend.hydra_inference.generate", fake_generate)
    monkeypatch.setattr("app.backend.hydra_inference.ModernBERTScorer", FakeScorer)

    result = get_robustness(
        model=MagicMock(),
        tokenizer=MagicMock(),
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        original_resp="Paris is the capital.",
        original_tokens=[],
        is_mcq=False,
        adv_suffix_bank=suffix_bank,
        bert_model=MagicMock(),
        bert_tokenizer=MagicMock(),
        device="cpu",
    )

    assert result["type"] == "nlp"
    assert result["attempts"] == 3
    assert result["flipped"] == 1
    assert result["worst_case"] is not None
    assert result["worst_case"]["suffix"] == "suffixB"
    assert result["worst_case"]["adv_response"] == "response B"
    assert result["worst_case"]["clean_response"] == "Paris is the capital."
    assert result["worst_case"]["flipped"] is True
    assert result["worst_case"]["score"] == pytest.approx(0.3)


def test_get_robustness_mcq_flip(monkeypatch):
    """MCQ path: first flipping suffix becomes worst_case; NLI is never invoked."""
    from app.backend.hydra_inference import get_robustness

    suffix_responses = {
        "suffixX": "A",  # no flip
        "suffixY": "B",  # flip -> worst case
        "suffixZ": "A",  # no flip
    }

    def fake_generate(model, tokenizer, messages, is_mcq, device):
        last = messages[-1]["content"]
        for suffix, resp in suffix_responses.items():
            if last.endswith(suffix):
                return (resp, [], [], 0.9)
        raise AssertionError(f"unexpected prompt: {last}")

    mock_scorer_cls = MagicMock()
    monkeypatch.setattr("app.backend.hydra_inference.generate", fake_generate)
    monkeypatch.setattr("app.backend.hydra_inference.ModernBERTScorer", mock_scorer_cls)

    result = get_robustness(
        model=MagicMock(),
        tokenizer=MagicMock(),
        messages=[{"role": "user", "content": "Which is correct?"}],
        original_resp="A",
        is_mcq=True,
        adv_suffix_bank=["suffixX", "suffixY", "suffixZ"],
        bert_model=MagicMock(),
        bert_tokenizer=MagicMock(),
        device="cpu",
    )

    mock_scorer_cls.assert_not_called()
    assert result["type"] == "mcq"
    assert result["attempts"] == 3
    assert result["flipped"] == 1
    assert result["worst_case"] is not None
    assert result["worst_case"]["suffix"] == "suffixY"
    assert result["worst_case"]["adv_response"] == "B"
    assert result["worst_case"]["clean_response"] == "A"
    assert result["worst_case"]["flipped"] is True
    assert result["worst_case"]["score"] is None


def test_load_hydra_returns_none_when_ensemble_load_fails():
    mock_tokenizer = MagicMock(spec=TokenizersBackend)

    with (
        patch("app.backend.hydra_inference.WEIGHTS_DIR", "fake_weights"),
        patch(
            "app.backend.hydra_inference.load_ensemble",
            side_effect=RuntimeError("missing manifest"),
        ),
        patch("app.backend.hydra_inference.AutoTokenizer") as mock_auto,
    ):
        mock_auto.from_pretrained.return_value = mock_tokenizer
        model, tok = load_hydra(device="cpu")

    assert model is None
    assert tok is None
