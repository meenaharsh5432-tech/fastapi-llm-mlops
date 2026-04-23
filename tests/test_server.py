import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

MOCK_GROQ_RESPONSE = {
    "choices": [{"message": {"content": "Use app.add_middleware(CORSMiddleware, ...)"}}],
    "usage": {"completion_tokens": 42},
}

def make_mock_post(*args, **kwargs):
    mock = MagicMock()
    mock.status_code = 200
    mock.json.return_value = MOCK_GROQ_RESPONSE
    return mock

with patch("requests.post", side_effect=make_mock_post):
    from server import app

client = TestClient(app)


def test_health_endpoint():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert "model_loaded" in data
    assert data["model_loaded"] is True


def test_root_endpoint():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["status"] == "running"


def test_chat_endpoint_structure():
    with patch("requests.post", side_effect=make_mock_post):
        r = client.post("/chat", json={"message": "How do I add CORS to FastAPI?"})
    assert r.status_code == 200
    data = r.json()
    assert "response" in data
    assert "model" in data
    assert "tokens_generated" in data
    assert "latency_ms" in data
