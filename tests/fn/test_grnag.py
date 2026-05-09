"""Tests for grnag.geron_nesterov_accelerated_gradient."""
import numpy as np
import pytest
from moirais.fn.grnag import geron_nesterov_accelerated_gradient


def test_grnag_basic():
    """Test basic functionality."""
    theta = 0.0
    grad_fn = (lambda v: v)
    v = np.random.default_rng(44).normal(0, 1, 100)
    eta = np.random.default_rng(42).normal(0, 1, 100)
    beta = 0.8
    result = geron_nesterov_accelerated_gradient(theta, grad_fn, v, eta, beta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grnag_edge():
    """Test edge cases."""
    theta = 0.0
    grad_fn = (lambda v: v)
    v = np.random.default_rng(44).normal(0, 1, 100)
    eta = np.random.default_rng(42).normal(0, 1, 100)
    beta = 0.8
    result = geron_nesterov_accelerated_gradient(theta, grad_fn, v, eta, beta)
    assert isinstance(result, dict)
