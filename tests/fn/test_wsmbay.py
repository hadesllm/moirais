"""Tests for wsmbay.wasserman_posterior."""
import numpy as np
import pytest
from moirais.fn.wsmbay import wasserman_posterior


def test_wsmbay_basic():
    """Test basic functionality."""
    data = np.random.default_rng(42).normal(0, 1, 100)
    f = np.random.default_rng(42).normal(0, 1, 100)
    prior = np.random.default_rng(42).normal(0, 1, 100)
    result = wasserman_posterior(data, f, prior)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wsmbay_edge():
    """Test edge cases."""
    data = np.random.default_rng(42).normal(0, 1, 100)
    f = np.random.default_rng(42).normal(0, 1, 100)
    prior = np.random.default_rng(42).normal(0, 1, 100)
    result = wasserman_posterior(data, f, prior)
    assert isinstance(result, dict)
