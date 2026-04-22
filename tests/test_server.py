import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

pytest.importorskip("torch")

from app.backend.server import app


@pytest.fixture
def client():
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()

    with (
        patch(
            "app.backend.server.load_hydra", return_value=(mock_model, mock_tokenizer)
        ),
        patch(
            "app.backend.server.load_bert", return_value=(mock_model, mock_tokenizer)
        ),
    ):
        with TestClient(app) as c:
            yield c


def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_analyse_returns_expected_shape(client):
    with (
        patch(
            "app.backend.server.generate",
            return_value=("Paris is the capital of France.", ["Paris", "is"], [], None),
        ),
        patch("app.backend.server.detect_mcq_bert", return_value=False),
    ):
        response = client.post(
            "/api/analyse",
            json={
                "messages": [
                    {"role": "user", "content": "What is the capital of France?"}
                ]
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert data["raw_response"] == "Paris is the capital of France."
    assert "claims" in data
    assert isinstance(data["claims"], list)
    assert "overall_confidence" in data
    assert data["security"]["certified"] is True
    assert data["security"]["tokens"] == ["Paris", "is"]
    assert data["security"]["resampled"] == []
    assert "tpa_budget" not in data["security"]
    assert "flagged_tokens" not in data["security"]
    assert "detail" not in data["security"]
    assert "robustness" in data
    assert "model" in data
    assert "is_mcq" in data
    assert isinstance(data["is_mcq"], bool) or data["is_mcq"] is None
    assert data["uncertainty"] == {"overall": None}


def test_analyse_surfaces_resampled_tokens(client):
    tokens = ["better", "answer"]
    resampled = [
        {"index": 0, "old_token": "worse", "new_token": "better", "severity": 1.0}
    ]
    with (
        patch(
            "app.backend.server.generate",
            return_value=("better answer", tokens, resampled, None),
        ),
        patch("app.backend.server.detect_mcq_bert", return_value=False),
    ):
        response = client.post(
            "/api/analyse",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )

    data = response.json()
    assert data["security"]["certified"] is True
    assert data["security"]["tokens"] == tokens
    assert data["security"]["resampled"] == resampled
    assert data["uncertainty"] == {"overall": None}


def test_analyse_fallback_security_on_hf(client):
    with patch("app.backend.server.detect_mcq_bert", return_value=False):
        with patch(
            "app.backend.server.call_hf_model", return_value="fallback response"
        ):
            response = client.post(
                "/api/analyse?hf=true",
                json={"messages": [{"role": "user", "content": "hi"}]},
            )

    data = response.json()
    assert data["security"]["certified"] is None
    assert data["security"]["tokens"] == []
    assert data["security"]["resampled"] == []
    assert data["uncertainty"] == {"overall": None}


def test_analyse_passes_full_message_history(client):
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "What is 2+2?"},
    ]
    with (
        patch(
            "app.backend.server.generate", return_value=("4.", ["4"], [], None)
        ) as mock_gen,
        patch("app.backend.server.detect_mcq_bert", return_value=False),
    ):
        client.post("/api/analyse", json={"messages": messages})

    called_messages = mock_gen.call_args[0][2]
    assert len(called_messages) == 3
    assert called_messages[-1]["content"] == "What is 2+2?"


def test_analyse_mcq_path_uses_bert_flag(client):
    with (
        patch(
            "app.backend.server.generate", return_value=("B", ["B"], [], 0.77)
        ) as mock_gen,
        patch("app.backend.server.detect_mcq_bert", return_value=True),
    ):
        response = client.post(
            "/api/analyse",
            json={"messages": [{"role": "user", "content": "A or B? A: foo B: bar"}]},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["raw_response"] == "B"
    assert data["is_mcq"] is True
    assert mock_gen.call_args.kwargs["is_mcq"] is True
    assert data["uncertainty"] == {"overall": 0.77}
