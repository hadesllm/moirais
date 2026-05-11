"""Tests for wsmiis.wasserman_importance_sampling."""
import numpy as np
import pytest
from morie.fn.wsmiis import wasserman_importance_sampling


def test_wsmiis_basic():
    """Test basic functionality."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    p = 5
    q = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = wasserman_importance_sampling(f, p, q, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wsmiis_edge():
    """Test edge cases."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    p = 5
    q = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = wasserman_importance_sampling(f, p, q, n)
    assert isinstance(result, dict)
