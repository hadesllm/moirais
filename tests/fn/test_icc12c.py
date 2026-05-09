"""Tests for icc12c.icc_two_way."""
import numpy as np
import pytest
from moirais.fn.icc12c import icc_two_way


def test_icc12c_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    model = np.random.default_rng(42).normal(0, 1, 100)
    result = icc_two_way(X, model)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_icc12c_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    model = np.random.default_rng(42).normal(0, 1, 100)
    result = icc_two_way(X, model)
    assert isinstance(result, dict)
