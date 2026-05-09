"""Tests for grnmfo.geron_nmf_objective."""
import numpy as np
import pytest
from moirais.fn.grnmfo import geron_nmf_objective


def test_grnmfo_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    W = np.random.default_rng(42).normal(0, 1, 100)
    H = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_nmf_objective(X, W, H)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grnmfo_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    W = np.random.default_rng(42).normal(0, 1, 100)
    H = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_nmf_objective(X, W, H)
    assert isinstance(result, dict)
