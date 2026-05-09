"""Tests for moirais.loc — local Ollama client."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from moirais.loc import LocalOllama, ModelInfo


# ---------------------------------------------------------------------------
# ModelInfo dataclass
# ---------------------------------------------------------------------------


def test_model_info_label_with_family():
    m = ModelInfo(name="gemma4:e2b", parameter_size="4B", family="gemma4")
    assert m.label == "Gemma4:4B"


def test_model_info_label_fallback():
    m = ModelInfo(name="custom-model:latest", size=3_000_000_000)
    assert "custom-model" in m.label


def test_model_info_size_gb():
    m = ModelInfo(name="x", size=4 * 1024**3)
    assert abs(m.size_gb - 4.0) < 0.01


# ---------------------------------------------------------------------------
# LocalOllama — construction
# ---------------------------------------------------------------------------


def test_default_model():
    client = LocalOllama()
    # Auto-detects from running Ollama; at minimum not empty if Ollama is running
    # If not running, falls back to empty string
    assert isinstance(client.model, str)


def test_override_model():
    client = LocalOllama(model="llama3.2:3b")
    assert client.model == "llama3.2:3b"


def test_env_override_model(monkeypatch):
    monkeypatch.setenv("MOIRAIS_OLLAMA_MODEL", "mistral:7b")
    client = LocalOllama()
    assert client.model == "mistral:7b"


def test_env_override_base_url(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://remote:11434")
    client = LocalOllama()
    assert client.base_url == "http://remote:11434"


# ---------------------------------------------------------------------------
# is_running
# ---------------------------------------------------------------------------


def test_is_running_true():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    with patch("moirais.loc.httpx.get", return_value=mock_resp):
        assert LocalOllama().is_running() is True


def test_is_running_false_connection_error():
    with patch("moirais.loc.httpx.get", side_effect=Exception("connection refused")):
        assert LocalOllama().is_running() is False


# ---------------------------------------------------------------------------
# list_models
# ---------------------------------------------------------------------------


_MOCK_TAGS = {
    "models": [
        {
            "name": "gemma4:e2b",
            "size": 2_500_000_000,
            "modified_at": "2026-03-30T12:00:00Z",
            "details": {
                "parameter_size": "4B",
                "family": "gemma4",
                "quantization_level": "Q4_K_M",
            },
        },
        {
            "name": "llama3.2:3b",
            "size": 1_900_000_000,
            "modified_at": "2026-03-30T11:00:00Z",
            "details": {
                "parameter_size": "3B",
                "family": "llama",
                "quantization_level": "Q4_K_M",
            },
        },
    ]
}


def _mock_get(url, **kwargs):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = _MOCK_TAGS
    resp.raise_for_status = MagicMock()
    return resp


def test_list_models():
    with patch("moirais.loc.httpx.get", side_effect=_mock_get):
        models = LocalOllama().list_models()
    assert len(models) == 2
    assert models[0].name == "gemma4:e2b"
    assert models[0].family == "gemma4"
    assert models[0].quantization == "Q4_K_M"


def test_model_names():
    with patch("moirais.loc.httpx.get", side_effect=_mock_get):
        names = LocalOllama().model_names()
    assert "gemma4:e2b" in names
    assert "llama3.2:3b" in names


def test_has_model():
    with patch("moirais.loc.httpx.get", side_effect=_mock_get):
        client = LocalOllama()
        assert client.has_model("gemma4:e2b") is True
        assert client.has_model("gemma4") is True  # prefix match
        assert client.has_model("phi:2b") is False


# ---------------------------------------------------------------------------
# chat
# ---------------------------------------------------------------------------


def test_chat():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {
        "message": {"role": "assistant", "content": "IPW is inverse probability weighting."},
        "done": True,
    }
    with patch("moirais.loc.httpx.post", return_value=mock_resp) as mock_post:
        client = LocalOllama(model="test-model")  # avoid auto-detect
        result = client.chat("What is IPW?")
    assert "IPW" in result
    call_args = mock_post.call_args
    body = call_args.kwargs.get("json") or call_args[1].get("json")
    assert body["model"]  # should be a non-empty model name
    assert body["stream"] is False


def test_chat_with_system_prompt():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"message": {"content": "ok"}, "done": True}
    with patch("moirais.loc.httpx.post", return_value=mock_resp) as mock_post:
        LocalOllama().chat("test", system="You are helpful")
    body = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json")
    messages = body["messages"]
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"


# ---------------------------------------------------------------------------
# remove
# ---------------------------------------------------------------------------


def test_remove_success():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    with patch("moirais.loc.httpx.delete", return_value=mock_resp):
        assert LocalOllama().remove("old-model") is True


def test_remove_not_found():
    mock_resp = MagicMock()
    mock_resp.status_code = 404
    with patch("moirais.loc.httpx.delete", return_value=mock_resp):
        assert LocalOllama().remove("nonexistent") is False


# ---------------------------------------------------------------------------
# build_messages helper
# ---------------------------------------------------------------------------


def test_build_messages_basic():
    msgs = LocalOllama._build_messages("hello", None, None)
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"


def test_build_messages_with_context():
    ctx = [{"role": "user", "content": "prior"}, {"role": "assistant", "content": "reply"}]
    msgs = LocalOllama._build_messages("new", "sys", ctx)
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert msgs[1]["content"] == "prior"
    assert msgs[-1]["content"] == "new"
