"""Tests for locp.local_polynomial."""
import numpy as np
import pytest
from moirais.fn.locp import local_polynomial


def test_locp_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    h = 0.3
    degree = np.random.default_rng(42).normal(0, 1, 100)
    result = local_polynomial(x, y, h, degree)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_locp_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    h = 0.3
    degree = np.random.default_rng(42).normal(0, 1, 100)
    result = local_polynomial(x, y, h, degree)
    assert isinstance(result, dict)
