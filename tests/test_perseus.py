from unittest.mock import patch

from moirais.perseus import agent_available, build_prompt, ask_percy


def test_build_prompt_includes_context():
    prompt = build_prompt("How should I test this?", context="Dataset: CPADS synthetic stub")

    assert "Context:" in prompt
    assert "Question:" in prompt
    assert "CPADS synthetic stub" in prompt


def test_agent_available_false_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)

    with patch("moirais.llm._probe_ollama", return_value=False), \
         patch("moirais.llm._probe_freeapi", return_value=False):
        assert agent_available() is False


def test_ask_percy_falls_back_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)

    with patch("moirais.llm._probe_ollama", return_value=False), \
         patch("moirais.llm._probe_freeapi", return_value=False):
        payload = ask_percy("Explain IPTW.")

    assert payload["mode"] == "local_fallback"
    assert "local agent mode" in payload["output_text"]
