"""Tests for matvar.matern_variogram_model."""
import numpy as np
import pytest
from morie.fn.matvar import matern_variogram_model


def test_matvar_basic():
    """Test basic functionality."""
    h = 0.3
    c0 = np.random.default_rng(42).normal(0, 1, 100)
    c = np.random.default_rng(42).normal(0, 1, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    nu = np.random.default_rng(42).normal(0, 1, 100)
    result = matern_variogram_model(h, c0, c, a, nu)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_matvar_edge():
    """Test edge cases."""
    h = 0.3
    c0 = np.random.default_rng(42).normal(0, 1, 100)
    c = np.random.default_rng(42).normal(0, 1, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    nu = np.random.default_rng(42).normal(0, 1, 100)
    result = matern_variogram_model(h, c0, c, a, nu)
    assert isinstance(result, dict)
