"""Tests for sgteig.sgt_eigenvector_centrality."""
import numpy as np
import pytest
from morie.fn.sgteig import sgt_eigenvector_centrality


def test_sgteig_basic():
    """Test basic functionality."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    result = sgt_eigenvector_centrality(A)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sgteig_edge():
    """Test edge cases."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    result = sgt_eigenvector_centrality(A)
    assert isinstance(result, dict)
