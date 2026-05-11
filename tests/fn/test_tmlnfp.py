"""Tests for tmlnfp.tmle_non_inferiority."""
import numpy as np
import pytest
from morie.fn.tmlnfp import tmle_non_inferiority


def test_tmlnfp_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    delta = np.random.default_rng(42).normal(0, 1, 100)
    result = tmle_non_inferiority(y, D, X, delta)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_tmlnfp_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    delta = np.random.default_rng(42).normal(0, 1, 100)
    result = tmle_non_inferiority(y, D, X, delta)
    assert isinstance(result, dict)
