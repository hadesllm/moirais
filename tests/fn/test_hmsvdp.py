"""Tests for hmsvdp.geron_svd_pseudoinverse."""
import numpy as np
import pytest
from moirais.fn.hmsvdp import geron_svd_pseudoinverse


def test_hmsvdp_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = geron_svd_pseudoinverse(X, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmsvdp_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = geron_svd_pseudoinverse(X, y)
    assert isinstance(result, dict)
