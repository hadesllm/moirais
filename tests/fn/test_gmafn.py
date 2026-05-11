"""Tests for morie.fn.gmafn — Gemma 4 native function calling."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from morie.fn.gmafn import gemma_function_call, _default_morie_tools, _execute_morie_function


def test_default_tools_structure():
    tools = _default_morie_tools()
    assert isinstance(tools, list)
    assert len(tools) >= 5
    for t in tools:
        assert t["type"] == "function"
        assert "name" in t["function"]
        assert "parameters" in t["function"]


def test_default_tools_contain_expected_functions():
    tools = _default_morie_tools()
    names = {t["function"]["name"] for t in tools}
    assert "dnorm" in names
    assert "ate" in names
    assert "luke" in names


def test_successful_chat_no_tool_calls():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {
        "message": {
            "role": "assistant",
            "content": "The normal density at x=0 is 0.3989.",
            "tool_calls": [],
        }
    }
    with patch("httpx.post", return_value=mock_resp):
        r = gemma_function_call("What is dnorm(0)?")
    assert r.name == "The normal density at x=0 is 0.3989."
    assert r.extra["tool_calls"] == []
    assert r.extra["model"] == "perseus:e2b"


def test_successful_chat_with_tool_call():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": {"name": "dnorm", "arguments": {"x": 0}}}
            ],
        }
    }
    with patch("httpx.post", return_value=mock_resp):
        with patch("morie.fn.gmafn._execute_morie_function", return_value=0.3989):
            r = gemma_function_call("compute dnorm(0)")
    assert r.name == "Function call(s) returned"
    assert len(r.extra["executed_results"]) == 1
    assert r.extra["executed_results"][0]["function"] == "dnorm"
    assert r.extra["executed_results"][0]["result"] == 0.3989


def test_connection_error():
    with patch("httpx.post", side_effect=Exception("connection refused")):
        r = gemma_function_call("test")
    assert "failed" in r.name.lower()
    assert "error" in r.extra


def test_custom_model_and_url():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"message": {"content": "ok", "tool_calls": []}}
    with patch("httpx.post", return_value=mock_resp) as mock_post:
        r = gemma_function_call(
            "test", model="gemma4:12b", base_url="http://remote:11434"
        )
    call_args = mock_post.call_args
    assert "remote:11434" in call_args[0][0]
    body = call_args[1]["json"]
    assert body["model"] == "gemma4:12b"


def test_execute_unknown_function():
    result = _execute_morie_function("nonexistent_fn_xyz", {})
    assert "error" in result


def test_registry_entry():
    from morie.fn._registry import REGISTRY
    assert "gmafn" in REGISTRY
    assert REGISTRY["gmafn"].category == "LLM"


def test_import_from_fn():
    from morie.fn.gmafn import gemma_function_call
    assert callable(gemma_function_call)
