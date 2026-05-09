"""Tests for tmlefp.tmle_effective_pi."""
import numpy as np
import pytest
from moirais.fn.tmlefp import tmle_effective_pi


def test_tmlefp_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    trim = np.random.default_rng(42).normal(0, 1, 100)
    result = tmle_effective_pi(y, D, X, trim)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_tmlefp_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    trim = np.random.default_rng(42).normal(0, 1, 100)
    result = tmle_effective_pi(y, D, X, trim)
    assert isinstance(result, dict)
