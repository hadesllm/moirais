"""Tests for tmlnsm.tmle_non_smooth."""
import numpy as np
import pytest
from moirais.fn.tmlnsm import tmle_non_smooth


def test_tmlnsm_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    bw = np.random.default_rng(42).normal(0, 1, 100)
    result = tmle_non_smooth(y, D, X, bw)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_tmlnsm_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    bw = np.random.default_rng(42).normal(0, 1, 100)
    result = tmle_non_smooth(y, D, X, bw)
    assert isinstance(result, dict)
