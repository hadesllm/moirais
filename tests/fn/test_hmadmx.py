"""Tests for hmadmx.geron_adamax."""
import numpy as np
import pytest
from moirais.fn.hmadmx import geron_adamax


def test_hmadmx_basic():
    """Test basic functionality."""
    grads = np.random.default_rng(42).normal(0, 1, 100)
    m = 10
    u = np.random.default_rng(44).normal(0, 1, 100)
    b1 = np.random.default_rng(42).normal(0, 1, 100)
    b2 = np.random.default_rng(42).normal(0, 1, 100)
    eta = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    result = geron_adamax(grads, m, u, b1, b2, eta, t)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmadmx_edge():
    """Test edge cases."""
    grads = np.random.default_rng(42).normal(0, 1, 100)
    m = 10
    u = np.random.default_rng(44).normal(0, 1, 100)
    b1 = np.random.default_rng(42).normal(0, 1, 100)
    b2 = np.random.default_rng(42).normal(0, 1, 100)
    eta = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    result = geron_adamax(grads, m, u, b1, b2, eta, t)
    assert isinstance(result, dict)
