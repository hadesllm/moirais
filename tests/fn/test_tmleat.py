"""Tests for tmleat.tmle_ate."""
import numpy as np
import pytest
from moirais.fn.tmleat import tmle_ate


def test_tmleat_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = tmle_ate(y, D, X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_tmleat_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = tmle_ate(y, D, X)
    assert isinstance(result, dict)
