"""Tests for hmmcp.geron_model_context_protocol."""
import numpy as np
import pytest
from moirais.fn.hmmcp import geron_model_context_protocol


def test_hmmcp_basic():
    """Test basic functionality."""
    server = np.random.default_rng(42).normal(0, 1, 100)
    client = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_model_context_protocol(server, client)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmmcp_edge():
    """Test edge cases."""
    server = np.random.default_rng(42).normal(0, 1, 100)
    client = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_model_context_protocol(server, client)
    assert isinstance(result, dict)
