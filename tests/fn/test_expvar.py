"""Tests for expvar.exponential_variogram_model."""
import numpy as np
import pytest
from moirais.fn.expvar import exponential_variogram_model


def test_expvar_basic():
    """Test basic functionality."""
    h = 0.3
    c0 = np.random.default_rng(42).normal(0, 1, 100)
    c = np.random.default_rng(42).normal(0, 1, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    result = exponential_variogram_model(h, c0, c, a)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_expvar_edge():
    """Test edge cases."""
    h = 0.3
    c0 = np.random.default_rng(42).normal(0, 1, 100)
    c = np.random.default_rng(42).normal(0, 1, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    result = exponential_variogram_model(h, c0, c, a)
    assert isinstance(result, dict)
