"""Tests for wassdt.wasserstein_1d."""
import numpy as np
import pytest
from moirais.fn.wassdt import wasserstein_1d


def test_wassdt_basic():
    """Test basic functionality."""
    p = 5
    q = np.random.default_rng(42).normal(0, 1, 100)
    support = np.random.default_rng(42).normal(0, 1, 100)
    result = wasserstein_1d(p, q, support)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wassdt_edge():
    """Test edge cases."""
    p = 5
    q = np.random.default_rng(42).normal(0, 1, 100)
    support = np.random.default_rng(42).normal(0, 1, 100)
    result = wasserstein_1d(p, q, support)
    assert isinstance(result, dict)
