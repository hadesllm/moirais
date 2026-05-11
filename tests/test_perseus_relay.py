"""Tests for perseus_relay — relay server and cloud client."""

from __future__ import annotations

import json
import threading
import time
from unittest.mock import patch

from morie.agent import AgentResponse, PerseusCloudAgent


class TestPerseusCloudAgent:

    @patch("morie.perseus_relay.PerseusCloudClient.ask")
    def test_chat_returns_agent_response(self, mock_ask):
        mock_ask.return_value = {
            "text": "Moran's I measures spatial autocorrelation.",
            "tool_calls": [{"name": "search_functions"}],
            "iterations": 2,
            "model": "perseus:e2b",
        }
        agent = PerseusCloudAgent("http://localhost:8421")
        resp = agent.chat("What is Moran's I?")
        assert isinstance(resp, AgentResponse)
        assert "Moran" in resp.text
        assert resp.iterations == 2
        assert len(resp.tool_calls_made) == 1

    @patch("morie.perseus_relay.PerseusCloudClient.ask")
    def test_chat_stream_yields_text(self, mock_ask):
        mock_ask.return_value = {
            "text": "The ATE is the average treatment effect.",
            "tool_calls": [],
            "iterations": 1,
            "model": "perseus:e2b",
        }
        agent = PerseusCloudAgent("http://localhost:8421")
        chunks = list(agent.chat_stream("What is ATE?"))
        full = "".join(chunks)
        assert "ATE" in full

    @patch("morie.perseus_relay.PerseusCloudClient.ask")
    def test_chat_stream_shows_tool_count(self, mock_ask):
        mock_ask.return_value = {
            "text": "Found ipw.",
            "tool_calls": [{"name": "search_functions"}, {"name": "run_morie_function"}],
            "iterations": 2,
            "model": "perseus:e2b",
        }
        agent = PerseusCloudAgent("http://localhost:8421")
        chunks = list(agent.chat_stream("Find propensity functions"))
        full = "".join(chunks)
        assert "2 tools called" in full


class TestPerseusCloudClient:

    def test_client_init(self):
        from morie.perseus_relay import PerseusCloudClient
        client = PerseusCloudClient("http://example.com:8421", token="secret")
        assert client.url == "http://example.com:8421"
        assert client.token == "secret"

    def test_client_strips_trailing_slash(self):
        from morie.perseus_relay import PerseusCloudClient
        client = PerseusCloudClient("http://example.com:8421/")
        assert client.url == "http://example.com:8421"


class TestCreateAgentCloud:

    def test_create_agent_with_cloud_url(self):
        from morie.agent import PerseusCloudAgent, create_agent
        agent = create_agent(cloud_url="http://example.com:8421")
        assert isinstance(agent, PerseusCloudAgent)

    @patch.dict("os.environ", {"PERSEUS_CLOUD_URL": "http://test:8421"})
    def test_create_agent_from_env(self):
        from morie.agent import PerseusCloudAgent, create_agent
        agent = create_agent()
        assert isinstance(agent, PerseusCloudAgent)


class TestRelayServer:

    def test_handler_import(self):
        from morie.perseus_relay import PerseusRelayHandler
        assert hasattr(PerseusRelayHandler, "do_POST")
        assert hasattr(PerseusRelayHandler, "do_GET")

    def test_serve_function_exists(self):
        from morie.perseus_relay import serve
        assert callable(serve)
