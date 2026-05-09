"""Tests for copExt.extremal_copula."""
import numpy as np
import pytest
from moirais.fn.copExt import extremal_copula


def test_copExt_basic():
    """Test basic functionality."""
    u = np.random.default_rng(44).normal(0, 1, 100)
    v = np.random.default_rng(44).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    result = extremal_copula(u, v, A)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_copExt_edge():
    """Test edge cases."""
    u = np.random.default_rng(44).normal(0, 1, 100)
    v = np.random.default_rng(44).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    result = extremal_copula(u, v, A)
    assert isinstance(result, dict)
