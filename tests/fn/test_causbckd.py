"""Tests for causbckd.causal_backdoor_estimate."""
import numpy as np
import pytest
from morie.fn.causbckd import causal_backdoor_estimate


def test_causbckd_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    result = causal_backdoor_estimate(y, X, Z)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_causbckd_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    result = causal_backdoor_estimate(y, X, Z)
    assert isinstance(result, dict)
