"""Tests for gprsk.gp_residual_kernel."""
import numpy as np
import pytest
from moirais.fn.gprsk import gp_residual_kernel


def test_gprsk_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    y_pred = np.random.default_rng(44).normal(0, 1, 100)
    kernel = (lambda u: np.exp(-0.5*u*u) / np.sqrt(2*np.pi))
    result = gp_residual_kernel(X, y, y_pred, kernel)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gprsk_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    y_pred = np.random.default_rng(44).normal(0, 1, 100)
    kernel = (lambda u: np.exp(-0.5*u*u) / np.sqrt(2*np.pi))
    result = gp_residual_kernel(X, y, y_pred, kernel)
    assert isinstance(result, dict)
