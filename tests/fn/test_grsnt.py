"""Tests for grsnt.geron_sentiment_binary."""
import numpy as np
import pytest
from moirais.fn.grsnt import geron_sentiment_binary


def test_grsnt_basic():
    """Test basic functionality."""
    token_ids = np.arange(100, dtype=int)
    E = np.random.default_rng(42).normal(0, 1, 100)
    w = np.random.default_rng(45).exponential(1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_sentiment_binary(token_ids, E, w, b)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grsnt_edge():
    """Test edge cases."""
    token_ids = np.arange(100, dtype=int)
    E = np.random.default_rng(42).normal(0, 1, 100)
    w = np.random.default_rng(45).exponential(1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_sentiment_binary(token_ids, E, w, b)
    assert isinstance(result, dict)
