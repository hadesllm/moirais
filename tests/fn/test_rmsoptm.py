"""Tests for rmsoptm.rmsprop."""
import numpy as np
import pytest
from moirais.fn.rmsoptm import rmsprop


def test_rmsoptm_basic():
    """Test basic functionality."""
    g = np.random.default_rng(43).normal(0, 1, 100)
    rho = 0.5
    lr = np.random.default_rng(42).normal(0, 1, 100)
    eps = np.random.default_rng(42).normal(0, 1, 100)
    result = rmsprop(g, rho, lr, eps)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rmsoptm_edge():
    """Test edge cases."""
    g = np.random.default_rng(43).normal(0, 1, 100)
    rho = 0.5
    lr = np.random.default_rng(42).normal(0, 1, 100)
    eps = np.random.default_rng(42).normal(0, 1, 100)
    result = rmsprop(g, rho, lr, eps)
    assert isinstance(result, dict)
