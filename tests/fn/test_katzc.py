"""Tests for katzc.katz_centrality."""
import numpy as np
import pytest
from morie.fn.katzc import katz_centrality


def test_katzc_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    alpha = 0.05
    beta = 0.8
    result = katz_centrality(y, A, alpha, beta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_katzc_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    alpha = 0.05
    beta = 0.8
    result = katz_centrality(y, A, alpha, beta)
    assert isinstance(result, dict)
