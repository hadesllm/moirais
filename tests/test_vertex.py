"""Tests for morie.vertex — mocked gcloud + httpx."""

import json
import subprocess
from unittest.mock import patch, MagicMock

import pytest

from morie import vertex


@pytest.fixture(autouse=True)
def _clear_token_cache():
    """Reset the module-level token cache between tests."""
    vertex._TOKEN_CACHE["token"] = None
    vertex._TOKEN_CACHE["expires_at"] = 0.0
    yield


def test_resolve_config_requires_project(monkeypatch):
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("MORIE_EE_PROJECT", raising=False)
    with pytest.raises(RuntimeError, match="GOOGLE_CLOUD_PROJECT"):
        vertex.resolve_config()


def test_resolve_config_picks_up_project(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("VERTEX_LOCATION", "us-west1")
    monkeypatch.setenv("VERTEX_MODEL", "gemini-2.5-pro")
    cfg = vertex.resolve_config()
    assert cfg.project == "test-project"
    assert cfg.location == "us-west1"
    assert cfg.model == "gemini-2.5-pro"


def test_resolve_config_falls_back_to_morie_ee_project(monkeypatch):
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.setenv("MORIE_EE_PROJECT", "ee-fallback-project")
    cfg = vertex.resolve_config()
    assert cfg.project == "ee-fallback-project"


def test_access_token_caches(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    cfg = vertex.resolve_config()
    with patch.object(vertex.subprocess, "run") as mock_run:
        mock_run.return_value = MagicMock(stdout="ya29.test.TOKEN\n", stderr="")
        t1 = vertex._access_token(cfg)
        t2 = vertex._access_token(cfg)
        assert t1 == "ya29.test.TOKEN"
        assert t2 == "ya29.test.TOKEN"
        # Called only once — second call hit the cache
        assert mock_run.call_count == 1


def test_access_token_propagates_gcloud_error(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    cfg = vertex.resolve_config()
    with patch.object(vertex.subprocess, "run") as mock_run:
        mock_run.side_effect = subprocess.CalledProcessError(
            1, "gcloud", stderr="reauth required"
        )
        with pytest.raises(RuntimeError, match="reauth required"):
            vertex._access_token(cfg)


def test_ask_gemini_sends_correct_payload(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    with patch.object(vertex, "_access_token", return_value="ya29.mock"), \
         patch.object(vertex._httpx, "Client") as mock_client_cls:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "candidates": [{
                "content": {
                    "parts": [{"text": "hello back"}],
                    "role": "model",
                },
            }],
        }
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post = MagicMock(return_value=mock_resp)
        mock_client_cls.return_value = mock_client

        out = vertex.ask_gemini("hello", system="be terse", temperature=0.3)
        assert out == "hello back"

        # Inspect the payload sent
        _, kwargs = mock_client.post.call_args
        payload = kwargs["json"]
        assert payload["contents"][0]["parts"][0]["text"] == "hello"
        assert payload["systemInstruction"]["parts"][0]["text"] == "be terse"
        assert payload["generationConfig"]["temperature"] == 0.3


def test_ask_gemini_raises_on_non_200(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    with patch.object(vertex, "_access_token", return_value="ya29.mock"), \
         patch.object(vertex._httpx, "Client") as mock_client_cls:
        mock_resp = MagicMock()
        mock_resp.status_code = 403
        mock_resp.text = "Permission denied on resource project/test-project"
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post = MagicMock(return_value=mock_resp)
        mock_client_cls.return_value = mock_client

        with pytest.raises(RuntimeError, match="403"):
            vertex.ask_gemini("hi")


def test_health_check_reports_ok(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    with patch.object(vertex, "ask_gemini", return_value="OK"):
        out = vertex.health_check()
    assert out["ok"] is True
    assert out["reply"] == "OK"
    assert out["project"] == "test-project"
    assert out["error"] is None


def test_health_check_captures_error(monkeypatch):
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("MORIE_EE_PROJECT", raising=False)
    out = vertex.health_check()
    assert out["ok"] is False
    assert "GOOGLE_CLOUD_PROJECT" in out["error"]
