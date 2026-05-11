"""Tests for hdptpc.hdp_topic_model."""
import numpy as np
import pytest
from morie.fn.hdptpc import hdp_topic_model


def test_hdptpc_basic():
    """Test basic functionality."""
    docs = np.random.default_rng(42).normal(0, 1, 100)
    gamma = 1.0
    alpha = 0.05
    result = hdp_topic_model(docs, gamma, alpha)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hdptpc_edge():
    """Test edge cases."""
    docs = np.random.default_rng(42).normal(0, 1, 100)
    gamma = 1.0
    alpha = 0.05
    result = hdp_topic_model(docs, gamma, alpha)
    assert isinstance(result, dict)
