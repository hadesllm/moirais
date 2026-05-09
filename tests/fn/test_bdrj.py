"""Tests for bdrj.backdoor_adjustment_formula."""
import numpy as np
import pytest
from moirais.fn.bdrj import backdoor_adjustment_formula


def test_bdrj_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    data = np.random.default_rng(42).normal(0, 1, 100)
    result = backdoor_adjustment_formula(X, Y, Z, data)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bdrj_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    data = np.random.default_rng(42).normal(0, 1, 100)
    result = backdoor_adjustment_formula(X, Y, Z, data)
    assert isinstance(result, dict)
