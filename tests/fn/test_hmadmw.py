"""Tests for hmadmw.geron_adamw."""
import numpy as np
import pytest
from morie.fn.hmadmw import geron_adamw


def test_hmadmw_basic():
    """Test basic functionality."""
    grads = np.random.default_rng(42).normal(0, 1, 100)
    m = 10
    v = np.random.default_rng(44).normal(0, 1, 100)
    b1 = np.random.default_rng(42).normal(0, 1, 100)
    b2 = np.random.default_rng(42).normal(0, 1, 100)
    eta = np.random.default_rng(42).normal(0, 1, 100)
    wd = np.random.default_rng(42).normal(0, 1, 100)
    eps = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    result = geron_adamw(grads, m, v, b1, b2, eta, wd, eps, t)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmadmw_edge():
    """Test edge cases."""
    grads = np.random.default_rng(42).normal(0, 1, 100)
    m = 10
    v = np.random.default_rng(44).normal(0, 1, 100)
    b1 = np.random.default_rng(42).normal(0, 1, 100)
    b2 = np.random.default_rng(42).normal(0, 1, 100)
    eta = np.random.default_rng(42).normal(0, 1, 100)
    wd = np.random.default_rng(42).normal(0, 1, 100)
    eps = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    result = geron_adamw(grads, m, v, b1, b2, eta, wd, eps, t)
    assert isinstance(result, dict)
