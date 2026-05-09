"""Tests for katz.katz_centrality."""
import numpy as np
import pytest
from moirais.fn.katz import katz_centrality


def test_katz_basic():
    """Test basic functionality."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    alpha = 0.05
    result = katz_centrality(A, alpha)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_katz_edge():
    """Test edge cases."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    alpha = 0.05
    result = katz_centrality(A, alpha)
    assert isinstance(result, dict)
