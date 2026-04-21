from unittest.mock import MagicMock, patch

from transformers import TokenizersBackend

from app.backend.hydra_inference import (
    _spans_from_poe_output,
    generate,
    load_hydra,
)


def test_generate_mcq_returns_letter():
    mock_model = MagicMock()
    mock_model.heads = [MagicMock()] * 9
    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template.return_value = "<prompt>"
    mock_tokenizer.encode.side_effect = lambda text, add_special_tokens=True: {
        "A": [100],
        "B": [101],
        "C": [102],
        "D": [103],
    }.get(text, [0])

    with patch(
        "app.backend.hydra_inference.poe_mcq_predict", return_value="B"
    ) as mock_mcq:
        raw, spans = generate(
            mock_model,
            mock_tokenizer,
            [{"role": "user", "content": "A or B?"}],
            is_mcq=True,
            device="cpu",
        )

    assert raw == "B"
    assert spans == []
    mock_mcq.assert_called_once()
    args, kwargs = mock_mcq.call_args
    assert args[0] is mock_model
    assert args[1] is mock_tokenizer
    assert args[3] == [100, 101, 102, 103]  # abcd token ids


def test_generate_nlp_returns_spans():
    mock_model = MagicMock()
    mock_model.heads = [MagicMock()] * 9
    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template.return_value = "<prompt>"
    mock_tokenizer.eos_token_id = 7
    mock_tokenizer.decode.return_value = "<eos>"

    output_parts = ["<chat-prefix>", "Hello", " world", "<eos>"]
    original_tokens = ["universe"]
    resampled_idxs = [2]  # " world" was resampled from "universe"

    with patch(
        "app.backend.hydra_inference.poe_generate_with_cache",
        return_value=(output_parts, original_tokens, resampled_idxs),
    ):
        raw, spans = generate(
            mock_model,
            mock_tokenizer,
            [{"role": "user", "content": "say hi"}],
            is_mcq=False,
            device="cpu",
        )

    assert raw == "Hello world"
    assert spans == [
        {"start": 5, "end": 11, "original": "universe", "replacement": " world"}
    ]


def test_spans_from_poe_output_no_resamples():
    mock_tokenizer = MagicMock()
    mock_tokenizer.eos_token_id = 7
    mock_tokenizer.decode.return_value = "<eos>"

    output_parts = ["<prefix>", "Hello", " world"]
    raw, spans = _spans_from_poe_output(mock_tokenizer, output_parts, [], [])

    assert raw == "Hello world"
    assert spans == []


def test_spans_from_poe_output_strips_trailing_eos():
    mock_tokenizer = MagicMock()
    mock_tokenizer.eos_token_id = 7
    mock_tokenizer.decode.return_value = "<eos>"

    output_parts = ["<prefix>", "Hi", "<eos>"]
    raw, spans = _spans_from_poe_output(mock_tokenizer, output_parts, [], [])

    assert raw == "Hi"
    assert spans == []


def test_spans_from_poe_output_drops_eos_resample():
    mock_tokenizer = MagicMock()
    mock_tokenizer.eos_token_id = 7
    mock_tokenizer.decode.return_value = "<eos>"

    # Resample landed on the trailing EOS; no visible span.
    output_parts = ["<prefix>", "Hi", "<eos>"]
    original_tokens = ["draft_eos"]
    resampled_idxs = [2]

    raw, spans = _spans_from_poe_output(
        mock_tokenizer, output_parts, original_tokens, resampled_idxs
    )

    assert raw == "Hi"
    assert spans == []


def test_load_hydra_returns_model_and_tokenizer():
    mock_model = MagicMock()
    mock_tokenizer = MagicMock(spec=TokenizersBackend)

    with (
        patch("app.backend.hydra_inference.WEIGHTS_DIR", "fake_weights"),
        patch(
            "app.backend.hydra_inference.load_ensemble",
            return_value=(mock_model, 9),
        ),
        patch("app.backend.hydra_inference.AutoTokenizer") as mock_auto,
    ):
        mock_auto.from_pretrained.return_value = mock_tokenizer
        result = load_hydra(device="cpu")

    assert len(result) == 2
    assert result[0] is mock_model
    assert result[1] is mock_tokenizer


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
