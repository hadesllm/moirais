"""Tests for hmhebb.geron_hebb_rule."""
import numpy as np
import pytest
from moirais.fn.hmhebb import geron_hebb_rule


def test_hmhebb_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    eta = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_hebb_rule(X, Y, eta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmhebb_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    eta = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_hebb_rule(X, Y, eta)
    assert isinstance(result, dict)
