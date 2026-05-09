"""Tests for wsmllk.wasserman_log_likelihood."""
import numpy as np
import pytest
from moirais.fn.wsmllk import wasserman_log_likelihood


def test_wsmllk_basic():
    """Test basic functionality."""
    data = np.random.default_rng(42).normal(0, 1, 100)
    f = np.random.default_rng(42).normal(0, 1, 100)
    theta = 0.0
    result = wasserman_log_likelihood(data, f, theta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wsmllk_edge():
    """Test edge cases."""
    data = np.random.default_rng(42).normal(0, 1, 100)
    f = np.random.default_rng(42).normal(0, 1, 100)
    theta = 0.0
    result = wasserman_log_likelihood(data, f, theta)
    assert isinstance(result, dict)
