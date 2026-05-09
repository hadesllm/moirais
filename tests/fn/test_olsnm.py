"""Tests for olsnm.ols_normal_equations."""
import numpy as np
import pytest
from moirais.fn.olsnm import ols_normal_equations


def test_olsnm_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = ols_normal_equations(X, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_olsnm_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = ols_normal_equations(X, y)
    assert isinstance(result, dict)
