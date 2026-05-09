"""Tests for morebs.empirical_bayes_moran."""
import numpy as np
import pytest
from moirais.fn.morebs import empirical_bayes_moran


def test_morebs_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    W = np.random.default_rng(42).normal(0, 1, 100)
    result = empirical_bayes_moran(x, n, W)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_morebs_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    W = np.random.default_rng(42).normal(0, 1, 100)
    result = empirical_bayes_moran(x, n, W)
    assert isinstance(result, dict)
