"""Tests for cgnonl.nonlinear_cg."""
import numpy as np
import pytest
from moirais.fn.cgnonl import nonlinear_cg


def test_cgnonl_basic():
    """Test basic functionality."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    grad_f = np.random.default_rng(42).normal(0, 1, 100)
    x0 = np.random.default_rng(42).normal(0, 1, 100)
    method = 'auto'
    result = nonlinear_cg(f, grad_f, x0, method)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_cgnonl_edge():
    """Test edge cases."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    grad_f = np.random.default_rng(42).normal(0, 1, 100)
    x0 = np.random.default_rng(42).normal(0, 1, 100)
    method = 'auto'
    result = nonlinear_cg(f, grad_f, x0, method)
    assert isinstance(result, dict)
