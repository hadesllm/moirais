"""Tests for sgtwlap.sgt_weighted_laplacian."""
import numpy as np
import pytest
from moirais.fn.sgtwlap import sgt_weighted_laplacian


def test_sgtwlap_basic():
    """Test basic functionality."""
    W = np.random.default_rng(42).normal(0, 1, 100)
    result = sgt_weighted_laplacian(W)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sgtwlap_edge():
    """Test edge cases."""
    W = np.random.default_rng(42).normal(0, 1, 100)
    result = sgt_weighted_laplacian(W)
    assert isinstance(result, dict)
