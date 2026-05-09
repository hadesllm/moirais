"""Tests for hmnag.geron_nesterov."""
import numpy as np
import pytest
from moirais.fn.hmnag import geron_nesterov


def test_hmnag_basic():
    """Test basic functionality."""
    grads = np.random.default_rng(42).normal(0, 1, 100)
    v = np.random.default_rng(44).normal(0, 1, 100)
    beta = 0.8
    eta = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_nesterov(grads, v, beta, eta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmnag_edge():
    """Test edge cases."""
    grads = np.random.default_rng(42).normal(0, 1, 100)
    v = np.random.default_rng(44).normal(0, 1, 100)
    beta = 0.8
    eta = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_nesterov(grads, v, beta, eta)
    assert isinstance(result, dict)
