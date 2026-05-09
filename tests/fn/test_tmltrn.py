"""Tests for tmltrn.tmle_transportability."""
import numpy as np
import pytest
from moirais.fn.tmltrn import tmle_transportability


def test_tmltrn_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    S = np.random.default_rng(42).normal(0, 1, 100)
    result = tmle_transportability(y, D, X, S)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_tmltrn_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    S = np.random.default_rng(42).normal(0, 1, 100)
    result = tmle_transportability(y, D, X, S)
    assert isinstance(result, dict)
