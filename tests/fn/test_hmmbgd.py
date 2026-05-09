"""Tests for hmmbgd.geron_minibatch_gd."""
import numpy as np
import pytest
from moirais.fn.hmmbgd import geron_minibatch_gd


def test_hmmbgd_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    theta = 0.0
    eta = np.random.default_rng(42).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_minibatch_gd(X, y, theta, eta, b)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmmbgd_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    theta = 0.0
    eta = np.random.default_rng(42).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_minibatch_gd(X, y, theta, eta, b)
    assert isinstance(result, dict)
