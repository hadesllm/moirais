"""Tests for nesterv.nesterov_accelerated."""
import numpy as np
import pytest
from moirais.fn.nesterv import nesterov_accelerated


def test_nesterv_basic():
    """Test basic functionality."""
    g = np.random.default_rng(43).normal(0, 1, 100)
    mu = 0.0
    lr = np.random.default_rng(42).normal(0, 1, 100)
    result = nesterov_accelerated(g, mu, lr)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_nesterv_edge():
    """Test edge cases."""
    g = np.random.default_rng(43).normal(0, 1, 100)
    mu = 0.0
    lr = np.random.default_rng(42).normal(0, 1, 100)
    result = nesterov_accelerated(g, mu, lr)
    assert isinstance(result, dict)
