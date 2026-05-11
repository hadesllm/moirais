"""Tests for morie.chat — interactive chat REPL and session management."""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

from morie.chat import (
    ChatMessage,
    ChatSession,
    SlashCommand,
    list_agents,
    load_agent_prompt,
)


# ---------------------------------------------------------------------------
# ChatMessage
# ---------------------------------------------------------------------------


class TestChatMessage:
    def test_defaults(self):
        msg = ChatMessage(role="user", content="hello")
        assert msg.role == "user"
        assert msg.content == "hello"
        assert msg.timestamp > 0

    def test_custom_timestamp(self):
        msg = ChatMessage(role="assistant", content="hi", timestamp=12345.0)
        assert msg.timestamp == 12345.0

    def test_provider_default_none(self):
        msg = ChatMessage(role="user", content="x")
        assert msg.provider is None


# ---------------------------------------------------------------------------
# ChatSession — slash commands
# ---------------------------------------------------------------------------


class TestChatSessionSlashCommands:
    def setup_method(self):
        self.session = ChatSession()

    def test_help_command(self):
        result = self.session.send("/help", stream=False)
        assert isinstance(result, str)
        assert "Available commands" in result
        assert "/list" in result
        assert "/quit" in result

    def test_list_command(self):
        result = self.session.send("/list", stream=False)
        assert isinstance(result, str)
        # Should contain at least one known module.
        assert "power-design" in result or "No modules" in result

    def test_doctor_command(self):
        result = self.session.send("/doctor", stream=False)
        assert isinstance(result, str)
        assert "MORIE Doctor" in result

    def test_agents_command(self):
        result = self.session.send("/agents", stream=False)
        assert isinstance(result, str)
        # Should find agents or report none found.
        assert "agents" in result.lower() or "Available" in result

    def test_provider_command(self):
        result = self.session.send("/provider", stream=False)
        assert isinstance(result, str)
        assert "provider" in result.lower()

    def test_clear_command(self):
        self.session.history.append(
            ChatMessage(role="user", content="test")
        )
        assert len(self.session.history) == 1
        result = self.session.send("/clear", stream=False)
        assert "cleared" in result.lower()
        assert len(self.session.history) == 0

    def test_history_empty(self):
        result = self.session.send("/history", stream=False)
        assert "No conversation" in result

    def test_history_with_messages(self):
        self.session.history.append(
            ChatMessage(role="user", content="what is IPW?")
        )
        result = self.session.send("/history", stream=False)
        assert "IPW" in result

    def test_unknown_command(self):
        result = self.session.send("/foobar", stream=False)
        assert "Unknown command" in result

    def test_quit_raises_eoferror(self):
        with pytest.raises(EOFError):
            self.session.send("/quit", stream=False)

    def test_exit_raises_eoferror(self):
        with pytest.raises(EOFError):
            self.session.send("/exit", stream=False)


# ---------------------------------------------------------------------------
# ChatSession — LLM interaction (mocked)
# ---------------------------------------------------------------------------


class TestChatSessionLLM:
    @patch("morie.chat.ask_multi", return_value="Test response")
    @patch("morie.chat.detect_available_provider", return_value="local")
    def test_send_message_non_stream(self, mock_provider, mock_ask):
        session = ChatSession()
        result = session.send("What is ATE?", stream=False)

        assert result == "Test response"
        assert len(session.history) == 2  # user + assistant
        assert session.history[0].role == "user"
        assert session.history[0].content == "What is ATE?"
        assert session.history[1].role == "assistant"
        assert session.history[1].content == "Test response"

    @patch("morie.chat.ask_multi", return_value=iter(["chunk1", "chunk2"]))
    @patch("morie.chat.detect_available_provider", return_value="local")
    def test_send_message_stream(self, mock_provider, mock_ask):
        session = ChatSession()
        result = session.send("What is ATE?", stream=True)

        # Should be an iterator.
        chunks = list(result)
        assert chunks == ["chunk1", "chunk2"]
        # History should be updated after consuming the stream.
        assert len(session.history) == 2
        assert session.history[1].content == "chunk1chunk2"

    @patch("morie.chat.ask_multi", return_value="Response 2")
    @patch("morie.chat.detect_available_provider", return_value="local")
    def test_multi_turn_history(self, mock_provider, mock_ask):
        session = ChatSession()

        # First turn.
        mock_ask.return_value = "First answer"
        session.send("First question", stream=False)

        # Second turn.
        mock_ask.return_value = "Second answer"
        session.send("Follow up", stream=False)

        assert len(session.history) == 4
        # Verify the messages array sent to ask_multi includes history.
        call_args = mock_ask.call_args_list[-1]
        messages = call_args[0][0]
        assert messages[0]["role"] == "system"
        # Should include previous turns.
        user_messages = [m for m in messages if m["role"] == "user"]
        assert len(user_messages) == 2


# ---------------------------------------------------------------------------
# ChatSession — agent loading
# ---------------------------------------------------------------------------


class TestChatSessionAgent:
    def test_agent_command_switch(self):
        session = ChatSession()
        original_prompt = session.system_prompt

        # Try switching to a nonexistent agent.
        result = session.send("/agent nonexistent-agent-xyz", stream=False)
        assert "not found" in result.lower()

    def test_agent_init(self):
        # Test with a nonexistent agent — should fall back to default.
        session = ChatSession(agent="nonexistent-agent-xyz")
        assert session.system_prompt  # Should have a default prompt.


# ---------------------------------------------------------------------------
# Agent listing
# ---------------------------------------------------------------------------


class TestListAgents:
    def test_list_agents_returns_list(self):
        agents = list_agents()
        assert isinstance(agents, list)
        # We know the repo has agents defined.
        for agent in agents:
            assert "name" in agent
            assert "description" in agent

    def test_load_agent_prompt_nonexistent(self):
        result = load_agent_prompt("nonexistent-agent-xyz-123")
        assert result is None


# ---------------------------------------------------------------------------
# Message building
# ---------------------------------------------------------------------------


class TestMessageBuilding:
    def test_build_messages_includes_system(self):
        session = ChatSession()
        session.history.append(ChatMessage(role="user", content="test"))
        messages = session._build_messages()

        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "test"

    def test_build_messages_preserves_order(self):
        session = ChatSession()
        session.history.append(ChatMessage(role="user", content="q1"))
        session.history.append(ChatMessage(role="assistant", content="a1"))
        session.history.append(ChatMessage(role="user", content="q2"))
        messages = session._build_messages()

        # system + 3 history messages
        assert len(messages) == 4
        assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
