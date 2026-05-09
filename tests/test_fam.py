"""Tests for moirais.fam — vendored OllamaFreeAPI client."""

import json
from unittest.mock import MagicMock, patch

import pytest

from moirais.fam import OllamaFreeAPI


class TestModelRegistry:
    """Test model loading from JSON files."""

    def test_init_loads_models(self):
        client = OllamaFreeAPI()
        # Should load from ollama_json/ directory
        families = client.list_families()
        assert isinstance(families, list)

    def test_list_models_returns_strings(self):
        client = OllamaFreeAPI()
        models = client.list_models()
        assert isinstance(models, list)
        for m in models:
            assert isinstance(m, str)

    def test_list_models_by_family(self):
        client = OllamaFreeAPI()
        families = client.list_families()
        if families:
            family_models = client.list_models(family=families[0])
            assert isinstance(family_models, list)

    def test_list_models_unknown_family(self):
        client = OllamaFreeAPI()
        models = client.list_models(family="nonexistent_family_xyz")
        assert models == []


class TestBuildPayload:
    def test_basic_payload(self):
        payload = OllamaFreeAPI._build_payload("model_a", "hello", stream=False)
        assert payload["model"] == "model_a"
        assert payload["prompt"] == "hello"
        assert payload["stream"] is False
        assert "options" in payload

    def test_payload_with_kwargs(self):
        payload = OllamaFreeAPI._build_payload(
            "model_a", "hello",
            temperature=0.1,
            num_predict=10000,
            stop=["\n"],
        )
        assert payload["options"]["temperature"] == 0.1
        assert payload["options"]["num_predict"] == 10000
        assert payload["options"]["stop"] == ["\n"]

    def test_stream_flag(self):
        payload = OllamaFreeAPI._build_payload("m", "p", stream=True)
        assert payload["stream"] is True


class TestGetModelInfo:
    def test_unknown_model_raises(self):
        client = OllamaFreeAPI()
        with pytest.raises(ValueError, match="not found"):
            client.get_model_info("totally_nonexistent_model_xyz_999")


class TestGetModelServers:
    def test_unknown_model_empty(self):
        client = OllamaFreeAPI()
        servers = client.get_model_servers("totally_nonexistent_model_xyz_999")
        assert servers == []


class TestChatNoServers:
    """Test chat/stream_chat error handling without actual network calls."""

    def test_chat_no_model_no_models_raises(self):
        client = OllamaFreeAPI()
        # Monkeypatch to have no models
        client._families = {}
        with pytest.raises(RuntimeError, match="No models available"):
            client.chat(prompt="hello")

    def test_stream_chat_no_models_raises(self):
        client = OllamaFreeAPI()
        client._families = {}
        with pytest.raises(RuntimeError, match="No models available"):
            list(client.stream_chat(prompt="hello"))

    def test_chat_no_servers_raises(self):
        client = OllamaFreeAPI()
        # Use a model name that won't have servers
        client._families = {"test": ["fake_model_no_servers"]}
        with pytest.raises(RuntimeError, match="No servers"):
            client.chat(prompt="hello", model="fake_model_no_servers")

    def test_stream_chat_no_servers_raises(self):
        client = OllamaFreeAPI()
        client._families = {"test": ["fake_model_no_servers"]}
        with pytest.raises(RuntimeError, match="No servers"):
            list(client.stream_chat(prompt="hello", model="fake_model_no_servers"))


class TestModelNameExtraction:
    def test_model_name_from_model_name_key(self):
        name = OllamaFreeAPI._model_name({"model_name": "foo"})
        assert name == "foo"

    def test_model_name_from_model_key(self):
        name = OllamaFreeAPI._model_name({"model": "bar"})
        assert name == "bar"

    def test_model_name_from_name_key(self):
        name = OllamaFreeAPI._model_name({"name": "baz"})
        assert name == "baz"

    def test_model_name_missing_returns_none(self):
        name = OllamaFreeAPI._model_name({})
        assert name is None


class TestExtractModels:
    def test_list_input(self):
        result = OllamaFreeAPI._extract_models([{"model": "a"}])
        assert result == [{"model": "a"}]

    def test_props_pageprops(self):
        data = {"props": {"pageProps": {"models": [{"model": "b"}]}}}
        result = OllamaFreeAPI._extract_models(data)
        assert result == [{"model": "b"}]

    def test_models_key(self):
        data = {"models": [{"model": "c"}]}
        result = OllamaFreeAPI._extract_models(data)
        assert result == [{"model": "c"}]
