"""Tests for admmop.admm."""
import numpy as np
import pytest
from moirais.fn.admmop import admm


def test_admmop_basic():
    """Test basic functionality."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    g = np.random.default_rng(43).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    c = np.random.default_rng(42).normal(0, 1, 100)
    rho = 0.5
    result = admm(f, g, A, B, c, rho)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_admmop_edge():
    """Test edge cases."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    g = np.random.default_rng(43).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    c = np.random.default_rng(42).normal(0, 1, 100)
    rho = 0.5
    result = admm(f, g, A, B, c, rho)
    assert isinstance(result, dict)
