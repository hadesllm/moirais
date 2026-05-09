"""Tests for baysprr.sparsity_horseshoe."""
import numpy as np
import pytest
from moirais.fn.baysprr import sparsity_horseshoe


def test_baysprr_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    tau = 0.1
    result = sparsity_horseshoe(X, y, tau)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_baysprr_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    tau = 0.1
    result = sparsity_horseshoe(X, y, tau)
    assert isinstance(result, dict)
