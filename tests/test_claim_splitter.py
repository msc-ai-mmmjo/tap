from unittest.mock import MagicMock, patch

from app.backend.claim_splitter import (
    _llm_decompose,
    _nltk_sentences,
    _parse_json_array,
    decompose_into_claims,
)


def test_parse_json_array_plain():
    assert _parse_json_array('["a", "b"]') == ["a", "b"]


def test_parse_json_array_extracts_from_prose():
    assert _parse_json_array('Sure, here you go: ["a", "b"] cheers.') == ["a", "b"]


def test_parse_json_array_handles_markdown_fence():
    raw = '```json\n["claim one", "claim two"]\n```'
    assert _parse_json_array(raw) == ["claim one", "claim two"]


def test_parse_json_array_strips_and_drops_empty():
    assert _parse_json_array('["  a  ", "", "b", "   "]') == ["a", "b"]


def test_parse_json_array_returns_none_when_no_brackets():
    assert _parse_json_array("no array here") is None


def test_parse_json_array_returns_none_on_malformed_json():
    assert _parse_json_array('["a",') is None


def test_parse_json_array_returns_none_on_non_list():
    assert _parse_json_array('{"x": 1}') is None


def test_parse_json_array_drops_non_string_entries():
    assert _parse_json_array('["a", 1, null, "b"]') == ["a", "b"]


def test_llm_decompose_returns_none_without_token():
    with patch("app.backend.claim_splitter.HF_TOKEN", ""):
        assert _llm_decompose("anything") is None


def test_llm_decompose_returns_none_when_client_raises():
    with (
        patch("app.backend.claim_splitter.HF_TOKEN", "tok"),
        patch("app.backend.claim_splitter.InferenceClient") as client_cls,
    ):
        client_cls.return_value.chat_completion.side_effect = RuntimeError("boom")
        assert _llm_decompose("input") is None


def test_llm_decompose_returns_parsed_claims_on_success():
    fake_response = MagicMock()
    fake_response.choices = [MagicMock()]
    fake_response.choices[0].message.content = '["claim 1", "claim 2"]'

    with (
        patch("app.backend.claim_splitter.HF_TOKEN", "tok"),
        patch("app.backend.claim_splitter.InferenceClient") as client_cls,
    ):
        client_cls.return_value.chat_completion.return_value = fake_response
        assert _llm_decompose("input") == ["claim 1", "claim 2"]


def test_llm_decompose_returns_none_on_unparseable_output():
    fake_response = MagicMock()
    fake_response.choices = [MagicMock()]
    fake_response.choices[0].message.content = "no json here, just prose"

    with (
        patch("app.backend.claim_splitter.HF_TOKEN", "tok"),
        patch("app.backend.claim_splitter.InferenceClient") as client_cls,
    ):
        client_cls.return_value.chat_completion.return_value = fake_response
        assert _llm_decompose("input") is None


def test_nltk_sentences_splits_two():
    out = _nltk_sentences("Hello world. How are you?")
    assert out == ["Hello world.", "How are you?"]


def test_nltk_sentences_strips_whitespace():
    out = _nltk_sentences("  One sentence.  ")
    assert out == ["One sentence."]


def test_nltk_sentences_falls_back_when_download_fails():
    with (
        patch("app.backend.claim_splitter.sent_tokenize", side_effect=LookupError),
        patch("app.backend.claim_splitter.nltk.download", return_value=False),
    ):
        assert _nltk_sentences("anything goes") == ["anything goes"]


def test_decompose_returns_empty_for_blank_input():
    assert decompose_into_claims("") == []
    assert decompose_into_claims("   \n  ") == []


def test_decompose_uses_llm_when_available():
    with patch(
        "app.backend.claim_splitter._llm_decompose", return_value=["a", "b"]
    ) as llm:
        assert decompose_into_claims("some response") == ["a", "b"]
        llm.assert_called_once()


def test_nltk_sentences_returns_text_when_tokenize_yields_nothing():
    with patch("app.backend.claim_splitter.sent_tokenize", return_value=["", "  "]):
        assert _nltk_sentences("fallback text") == ["fallback text"]


def test_decompose_falls_back_when_llm_returns_empty_list():
    with (
        patch("app.backend.claim_splitter._llm_decompose", return_value=[]),
        patch(
            "app.backend.claim_splitter._nltk_sentences",
            return_value=["sentence."],
        ) as nltk_split,
    ):
        assert decompose_into_claims("response") == ["sentence."]
        nltk_split.assert_called_once()


def test_decompose_falls_back_to_nltk_when_llm_returns_none():
    with (
        patch("app.backend.claim_splitter._llm_decompose", return_value=None),
        patch(
            "app.backend.claim_splitter._nltk_sentences",
            return_value=["one.", "two."],
        ) as nltk_split,
    ):
        assert decompose_into_claims("response") == ["one.", "two."]
        nltk_split.assert_called_once()
