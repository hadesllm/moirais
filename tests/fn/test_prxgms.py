"""Tests for prxgms.proximal_gradient_method."""
import numpy as np
import pytest
from morie.fn.prxgms import proximal_gradient_method


def test_prxgms_basic():
    """Test basic functionality."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    grad_f = np.random.default_rng(42).normal(0, 1, 100)
    x0 = np.random.default_rng(42).normal(0, 1, 100)
    lr = np.random.default_rng(42).normal(0, 1, 100)
    lam = 0.1
    result = proximal_gradient_method(f, grad_f, x0, lr, lam)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_prxgms_edge():
    """Test edge cases."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    grad_f = np.random.default_rng(42).normal(0, 1, 100)
    x0 = np.random.default_rng(42).normal(0, 1, 100)
    lr = np.random.default_rng(42).normal(0, 1, 100)
    lam = 0.1
    result = proximal_gradient_method(f, grad_f, x0, lr, lam)
    assert isinstance(result, dict)
