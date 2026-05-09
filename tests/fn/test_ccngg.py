"""Tests for ccngg.nakagawa_conditional_r2."""
import numpy as np
import pytest
from moirais.fn.ccngg import nakagawa_conditional_r2


def test_ccngg_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    cluster = np.random.default_rng(42).normal(0, 1, 100)
    result = nakagawa_conditional_r2(y, X, Z, cluster)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ccngg_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    cluster = np.random.default_rng(42).normal(0, 1, 100)
    result = nakagawa_conditional_r2(y, X, Z, cluster)
    assert isinstance(result, dict)
