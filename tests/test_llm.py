"""Tests for the MORIE LLM integration layer (morie.llm).

Covers:
- Provider detection fallback logic
- Context building
- Local fallback text generation
- Message construction
- ask() routing under mocked conditions
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from morie.llm import (
    _PROVIDER_API,
    _PROVIDER_LOCAL,
    _PROVIDER_OLLAMA,
    _PROVIDER_OPENAI,
    _build_messages,
    _format_context_block,
    _local_fallback,
    ask,
    agent_available,
    assistant_available,
    build_morie_context,
    detect_available_provider,
)


# ---------------------------------------------------------------------------
# Provider detection
# ---------------------------------------------------------------------------


class TestDetectAvailableProvider:
    """Test the provider detection chain: ollama -> api -> openai -> local."""

    def test_ollama_detected_when_running(self, monkeypatch):
        """When Ollama health check passes, provider should be 'ollama'."""
        monkeypatch.delenv("LLM_API_BASE_URL", raising=False)
        monkeypatch.delenv("LLM_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        with patch("morie.llm._probe_ollama", return_value=True):
            assert detect_available_provider() == _PROVIDER_OLLAMA

    def test_api_detected_when_configured(self, monkeypatch):
        """When Ollama is down but LLM_API_BASE_URL and key are set."""
        monkeypatch.setenv("LLM_API_BASE_URL", "https://example.com/v1")
        monkeypatch.setenv("LLM_API_KEY", "test-key-123")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        with patch("morie.llm._probe_ollama", return_value=False), \
             patch("morie.llm._probe_freeapi", return_value=False):
            assert detect_available_provider() == _PROVIDER_API

    def test_openai_detected_when_key_set(self, monkeypatch):
        """When Ollama is down and no generic API, but OPENAI_API_KEY is set."""
        monkeypatch.delenv("LLM_API_BASE_URL", raising=False)
        monkeypatch.delenv("LLM_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")

        with patch("morie.llm._probe_ollama", return_value=False), \
             patch("morie.llm._probe_freeapi", return_value=False):
            assert detect_available_provider() == _PROVIDER_OPENAI

    def test_local_when_nothing_configured(self, monkeypatch):
        """When no provider is available, detection returns 'local'."""
        monkeypatch.delenv("LLM_API_BASE_URL", raising=False)
        monkeypatch.delenv("LLM_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)

        with patch("morie.llm._probe_ollama", return_value=False), \
             patch("morie.llm._probe_freeapi", return_value=False):
            assert detect_available_provider() == _PROVIDER_LOCAL

    def test_ollama_takes_priority_over_api(self, monkeypatch):
        """Even with API keys set, Ollama should be preferred if available."""
        monkeypatch.setenv("LLM_API_BASE_URL", "https://example.com/v1")
        monkeypatch.setenv("LLM_API_KEY", "test-key")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        with patch("morie.llm._probe_ollama", return_value=True):
            assert detect_available_provider() == _PROVIDER_OLLAMA

    def test_api_takes_priority_over_openai(self, monkeypatch):
        """With Ollama down, generic API should be preferred over OpenAI."""
        monkeypatch.setenv("LLM_API_BASE_URL", "https://example.com/v1")
        monkeypatch.setenv("LLM_API_KEY", "test-key")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        with patch("morie.llm._probe_ollama", return_value=False), \
             patch("morie.llm._probe_freeapi", return_value=False):
            assert detect_available_provider() == _PROVIDER_API

    def test_empty_string_env_vars_ignored(self, monkeypatch):
        """Empty-string env vars should not count as configured."""
        monkeypatch.setenv("LLM_API_BASE_URL", "")
        monkeypatch.setenv("LLM_API_KEY", "")
        monkeypatch.setenv("OPENAI_API_KEY", "  ")

        with patch("morie.llm._probe_ollama", return_value=False), \
             patch("morie.llm._probe_freeapi", return_value=False):
            assert detect_available_provider() == _PROVIDER_LOCAL


# ---------------------------------------------------------------------------
# Context building
# ---------------------------------------------------------------------------


class TestBuildMorieContext:
    """Test the context dictionary builder."""

    def test_returns_required_keys(self):
        ctx = build_morie_context()
        assert "module_list" in ctx
        assert "cpads_schema" in ctx
        assert "cwd" in ctx
        assert "repo_root" in ctx

    def test_module_list_is_nonempty(self):
        ctx = build_morie_context()
        assert len(ctx["module_list"]) > 0
        first = ctx["module_list"][0]
        assert "name" in first
        assert "description" in first

    def test_cpads_schema_has_required_variables(self):
        ctx = build_morie_context()
        schema = ctx["cpads_schema"]
        assert "required_variables" in schema
        assert "weight" in schema["required_variables"]

    def test_custom_repo_root(self, tmp_path):
        ctx = build_morie_context(repo_root=str(tmp_path))
        assert ctx["repo_root"] == str(tmp_path)

    def test_cwd_is_current_directory(self):
        ctx = build_morie_context()
        assert ctx["cwd"] == os.getcwd()


class TestFormatContextBlock:
    """Test the context-to-text formatter."""

    def test_none_returns_empty(self):
        assert _format_context_block(None) == ""

    def test_empty_dict_returns_empty(self):
        assert _format_context_block({}) == ""

    def test_includes_module_names(self):
        context = {
            "module_list": [
                {"name": "data-wrangling", "description": "Clean data"},
                {"name": "causal-estimators", "description": "Causal"},
            ]
        }
        block = _format_context_block(context)
        assert "data-wrangling" in block
        assert "causal-estimators" in block

    def test_includes_cpads_variables(self):
        context = {
            "cpads_schema": {"required_variables": ["weight", "gender"]}
        }
        block = _format_context_block(context)
        assert "weight" in block
        assert "gender" in block


# ---------------------------------------------------------------------------
# Local fallback
# ---------------------------------------------------------------------------


class TestLocalFallback:
    """Test the static local fallback response."""

    def test_basic_fallback_contains_morie(self):
        result = _local_fallback("Hello")
        assert "MORIE" in result
        assert "local-only mode" in result

    def test_cpads_keyword_enriches_response(self):
        result = _local_fallback("Tell me about CPADS data")
        assert "CPADS data contract" in result
        assert "weight" in result

    def test_ipw_keyword_enriches_response(self):
        result = _local_fallback("How do I use IPW?")
        assert "propensity-scores" in result

    def test_module_keyword_enriches_response(self):
        result = _local_fallback("List all modules")
        assert "data-wrangling" in result

    def test_no_enrichment_for_unrelated_prompt(self):
        result = _local_fallback("What is the meaning of life?")
        assert "CPADS data contract" not in result
        assert "propensity-scores" not in result


# ---------------------------------------------------------------------------
# Message building
# ---------------------------------------------------------------------------


class TestBuildMessages:
    """Test the chat message array construction."""

    def test_default_messages_have_system_and_user(self):
        msgs = _build_messages("What is ATE?")
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert "ATE" in msgs[1]["content"]

    def test_custom_system_prompt(self):
        msgs = _build_messages("test", system_prompt="Custom system")
        assert msgs[0]["content"] == "Custom system"

    def test_context_injected_into_system(self):
        context = {
            "module_list": [{"name": "test-module", "description": "A test"}],
        }
        msgs = _build_messages("question", context=context)
        assert "test-module" in msgs[0]["content"]


# ---------------------------------------------------------------------------
# ask() routing
# ---------------------------------------------------------------------------


class TestAsk:
    """Test the main ask() function routing and fallback behavior."""

    def test_local_fallback_when_no_provider(self, monkeypatch):
        monkeypatch.delenv("LLM_API_BASE_URL", raising=False)
        monkeypatch.delenv("LLM_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        with patch("morie.llm._probe_ollama", return_value=False), \
             patch("morie.llm._probe_freeapi", return_value=False):
            result = ask("What is TMLE?")
            assert isinstance(result, str)
            assert "MORIE" in result
            assert "local-only mode" in result

    def test_local_fallback_stream_returns_iterator(self, monkeypatch):
        monkeypatch.delenv("LLM_API_BASE_URL", raising=False)
        monkeypatch.delenv("LLM_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        with patch("morie.llm._probe_ollama", return_value=False), \
             patch("morie.llm._probe_freeapi", return_value=False):
            result = ask("What is TMLE?", stream=True)
            chunks = list(result)
            assert len(chunks) == 1
            assert "MORIE" in chunks[0]

    def test_forced_local_provider(self):
        result = ask("test", provider="local")
        assert isinstance(result, str)
        assert "MORIE" in result

    def test_ollama_request_made_when_detected(self, monkeypatch):
        """Verify that ask() calls the Ollama endpoint when detected."""
        monkeypatch.delenv("LLM_API_BASE_URL", raising=False)
        monkeypatch.delenv("LLM_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {"message": {"content": "TMLE is a semiparametric estimator."}}
            ]
        }

        with (
            patch("morie.llm._probe_ollama", return_value=True),
            patch("morie.llm._request_completion", return_value=mock_response) as mock_req,
        ):
            result = ask("What is TMLE?")
            assert result == "TMLE is a semiparametric estimator."
            # Verify call was made to Ollama base URL
            call_args = mock_req.call_args
            assert "localhost:11434" in call_args[0][0]

    def test_fallback_chain_on_ollama_failure(self, monkeypatch):
        """When Ollama is detected but the request fails, fall through to API."""
        monkeypatch.setenv("LLM_API_BASE_URL", "https://example.com/v1")
        monkeypatch.setenv("LLM_API_KEY", "test-key")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        mock_response_ok = MagicMock()
        mock_response_ok.status_code = 200
        mock_response_ok.raise_for_status = MagicMock()
        mock_response_ok.json.return_value = {
            "choices": [
                {"message": {"content": "Answer from API"}}
            ]
        }

        call_count = 0

        def side_effect(base_url, model, messages, **kwargs):
            nonlocal call_count
            call_count += 1
            if "localhost" in base_url:
                raise ConnectionError("Ollama down")
            return mock_response_ok

        with (
            patch("morie.llm._probe_ollama", return_value=True),
            patch("morie.llm._request_completion", side_effect=side_effect),
        ):
            result = ask("What is AIPW?")
            assert result == "Answer from API"
            assert call_count == 2  # Ollama attempt + API attempt


# ---------------------------------------------------------------------------
# agent_available()
# ---------------------------------------------------------------------------


class TestAgentAvailable:
    """Test the agent_available() function and its backward-compat alias."""

    def test_false_when_nothing_configured(self, monkeypatch):
        monkeypatch.delenv("LLM_API_BASE_URL", raising=False)
        monkeypatch.delenv("LLM_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        with patch("morie.llm._probe_ollama", return_value=False), \
             patch("morie.llm._probe_freeapi", return_value=False):
            assert agent_available() is False
            assert assistant_available() is False

    def test_true_when_ollama_available(self, monkeypatch):
        monkeypatch.delenv("LLM_API_BASE_URL", raising=False)
        monkeypatch.delenv("LLM_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        with patch("morie.llm._probe_ollama", return_value=True):
            assert agent_available() is True

    def test_true_when_openai_key_set(self, monkeypatch):
        monkeypatch.delenv("LLM_API_BASE_URL", raising=False)
        monkeypatch.delenv("LLM_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        with patch("morie.llm._probe_ollama", return_value=False), \
             patch("morie.llm._probe_freeapi", return_value=False):
            assert agent_available() is True
