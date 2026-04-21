from unittest.mock import MagicMock, patch

import torch
from transformers import TokenizersBackend

from app.backend.hydra_inference import generate


def test_generate_returns_decoded_tokens():
    vocab_size = 10
    logits = torch.zeros(1, 1, 1, vocab_size)
    logits[0, 0, 0, 7] = 10.0

    mock_model = MagicMock()
    mock_model.return_value = logits

    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template.return_value = "<|user|>hi<|end|>"
    mock_tokenizer.encode.return_value = [1, 2, 3]
    mock_tokenizer.decode.return_value = "hello world"

    result = generate(
        mock_model,
        mock_tokenizer,
        [{"role": "user", "content": "hi"}],
        max_new_tokens=3,
        device="cpu",
    )

    assert result == "hello world"
    mock_tokenizer.decode.assert_called_once_with([7, 7, 7], skip_special_tokens=True)


def test_generate_calls_reset_kv_cache():
    vocab_size = 10
    logits = torch.zeros(1, 1, 1, vocab_size)
    logits[0, 0, 0, 2] = 10.0

    mock_model = MagicMock()
    mock_model.return_value = logits

    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template.return_value = "<prompt>"
    mock_tokenizer.encode.return_value = [10, 20]
    mock_tokenizer.decode.return_value = "ok"

    generate(
        mock_model,
        mock_tokenizer,
        [{"role": "user", "content": "test"}],
        max_new_tokens=5,
        device="cpu",
    )

    mock_model.reset_kv_cache.assert_called_once_with()
    mock_model.init_kv_cache.assert_not_called()


def test_generate_stops_at_eos():
    vocab_size = 10
    logits = torch.zeros(1, 1, 1, vocab_size)
    logits[0, 0, 0, 7] = 10.0  # token 7 is always selected

    mock_model = MagicMock()
    mock_model.return_value = logits

    mock_tokenizer = MagicMock()
    mock_tokenizer.eos_token_id = 7  # EOS is token 7
    mock_tokenizer.apply_chat_template.return_value = "<prompt>"
    mock_tokenizer.encode.return_value = [1, 2]
    mock_tokenizer.decode.return_value = ""

    generate(
        mock_model,
        mock_tokenizer,
        [{"role": "user", "content": "hi"}],
        max_new_tokens=10,
        device="cpu",
    )

    # only 1 token generated before EOS stops the loop
    mock_tokenizer.decode.assert_called_once_with([7], skip_special_tokens=True)
    assert mock_model.call_count == 1


def test_load_model_returns_model_and_tokenizer():
    from app.backend.hydra_inference import load_hydra

    mock_model = MagicMock()
    mock_tokenizer = MagicMock(spec=TokenizersBackend)

    with (
        patch("app.backend.hydra_inference.WEIGHTS_DIR", "fake_weights"),
        patch("app.backend.hydra_inference.build_base_model", return_value=mock_model),
        patch("app.backend.hydra_inference.AutoTokenizer") as mock_auto,
    ):
        mock_auto.from_pretrained.return_value = mock_tokenizer
        result = load_hydra(device="cpu")

    assert len(result) == 2
    assert result[0] is mock_model
    assert result[1] is mock_tokenizer
