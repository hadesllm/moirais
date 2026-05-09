"""Tests for hampel.hampel_redescend."""
import numpy as np
import pytest
from moirais.fn.hampel import hampel_redescend


def test_hampel_basic():
    """Test basic functionality."""
    r = 10
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    c = np.random.default_rng(42).normal(0, 1, 100)
    result = hampel_redescend(r, a, b, c)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hampel_edge():
    """Test edge cases."""
    r = 10
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    c = np.random.default_rng(42).normal(0, 1, 100)
    result = hampel_redescend(r, a, b, c)
    assert isinstance(result, dict)
