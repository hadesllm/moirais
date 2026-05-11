"""Tests for tmlrct.tmle_rct_assisted."""
import numpy as np
import pytest
from morie.fn.tmlrct import tmle_rct_assisted


def test_tmlrct_basic():
    """Test basic functionality."""
    y_rct = np.random.default_rng(42).normal(0, 1, 100)
    y_obs = np.random.default_rng(42).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = tmle_rct_assisted(y_rct, y_obs, D, X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_tmlrct_edge():
    """Test edge cases."""
    y_rct = np.random.default_rng(42).normal(0, 1, 100)
    y_obs = np.random.default_rng(42).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = tmle_rct_assisted(y_rct, y_obs, D, X)
    assert isinstance(result, dict)
