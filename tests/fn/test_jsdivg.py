"""Tests for jsdivg.jensen_shannon_divergence."""
import numpy as np
import pytest
from moirais.fn.jsdivg import jensen_shannon_divergence


def test_jsdivg_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    p = 5
    q = np.random.default_rng(42).normal(0, 1, 100)
    result = jensen_shannon_divergence(y, p, q)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_jsdivg_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    p = 5
    q = np.random.default_rng(42).normal(0, 1, 100)
    result = jensen_shannon_divergence(y, p, q)
    assert isinstance(result, dict)
