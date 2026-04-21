import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

pytest.importorskip("torch")

from app.backend.server import app


@pytest.fixture
def client():
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()

    with patch(
        "app.backend.server.load_hydra", return_value=(mock_model, mock_tokenizer)
    ), patch(
        "app.backend.server.load_bert", return_value=(mock_model, mock_tokenizer)
    ):
        with TestClient(app) as c:
            yield c


def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_analyse_returns_expected_shape(client):
    with patch(
        "app.backend.server.generate", return_value=("Paris is the capital of France.", False)
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
    if data["claims"]:
        claim = data["claims"][0]
        assert "text" in claim
        assert "confidence" in claim
        assert "confidence_level" in claim
        assert "guidance" in claim
    assert "overall_confidence" in data
    assert "security" in data
    assert "robustness" in data
    assert "model" in data
    assert "is_mcq" in data
    assert isinstance(data["is_mcq"], bool) or data["is_mcq"] is None


def test_analyse_passes_full_message_history(client):
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "What is 2+2?"},
    ]
    with patch("app.backend.server.generate", return_value=("4.", False)) as mock_gen:
        client.post("/api/analyse", json={"messages": messages})

    called_messages = mock_gen.call_args[0][2]  # 3rd positional arg is messages
    assert len(called_messages) == 3
    assert called_messages[-1]["content"] == "What is 2+2?"
