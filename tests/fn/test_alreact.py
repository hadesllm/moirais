"""Tests for alreact.alammar_react_agent_loop."""
import numpy as np
import pytest
from moirais.fn.alreact import alammar_react_agent_loop


def test_alreact_basic():
    """Test basic functionality."""
    query = np.random.default_rng(42).normal(0, 1, 100)
    tools = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    max_steps = np.random.default_rng(42).normal(0, 1, 100)
    result = alammar_react_agent_loop(query, tools, model, max_steps)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_alreact_edge():
    """Test edge cases."""
    query = np.random.default_rng(42).normal(0, 1, 100)
    tools = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    max_steps = np.random.default_rng(42).normal(0, 1, 100)
    result = alammar_react_agent_loop(query, tools, model, max_steps)
    assert isinstance(result, dict)
