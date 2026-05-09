"""Tests for timesnet.timesnet."""
import numpy as np
import pytest
from moirais.fn.timesnet import timesnet


def test_timesnet_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    top_k = np.random.default_rng(42).normal(0, 1, 100)
    result = timesnet(X, top_k)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_timesnet_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    top_k = np.random.default_rng(42).normal(0, 1, 100)
    result = timesnet(X, top_k)
    assert isinstance(result, dict)
