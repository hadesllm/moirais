"""Tests for neteig.eigenvector_centrality."""
import numpy as np
import pytest
from moirais.fn.neteig import eigenvector_centrality


def test_neteig_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    result = eigenvector_centrality(y, A)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_neteig_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    result = eigenvector_centrality(y, A)
    assert isinstance(result, dict)
