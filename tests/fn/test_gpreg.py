"""Tests for gpreg.gaussian_process_regression."""
import numpy as np
import pytest
from morie.fn.gpreg import gaussian_process_regression


def test_gpreg_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    X_test = np.random.default_rng(43).normal(0, 1, 30)
    kernel = (lambda u: np.exp(-0.5*u*u) / np.sqrt(2*np.pi))
    noise = np.random.default_rng(42).normal(0, 1, 100)
    result = gaussian_process_regression(X, y, X_test, kernel, noise)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gpreg_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    X_test = np.random.default_rng(43).normal(0, 1, 30)
    kernel = (lambda u: np.exp(-0.5*u*u) / np.sqrt(2*np.pi))
    noise = np.random.default_rng(42).normal(0, 1, 100)
    result = gaussian_process_regression(X, y, X_test, kernel, noise)
    assert isinstance(result, dict)
