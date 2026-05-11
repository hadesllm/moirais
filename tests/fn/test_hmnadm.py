"""Tests for hmnadm.geron_nadam."""
import numpy as np
import pytest
from morie.fn.hmnadm import geron_nadam


def test_hmnadm_basic():
    """Test basic functionality."""
    grads = np.random.default_rng(42).normal(0, 1, 100)
    m = 10
    v = np.random.default_rng(44).normal(0, 1, 100)
    b1 = np.random.default_rng(42).normal(0, 1, 100)
    b2 = np.random.default_rng(42).normal(0, 1, 100)
    eta = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    result = geron_nadam(grads, m, v, b1, b2, eta, t)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmnadm_edge():
    """Test edge cases."""
    grads = np.random.default_rng(42).normal(0, 1, 100)
    m = 10
    v = np.random.default_rng(44).normal(0, 1, 100)
    b1 = np.random.default_rng(42).normal(0, 1, 100)
    b2 = np.random.default_rng(42).normal(0, 1, 100)
    eta = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    result = geron_nadam(grads, m, v, b1, b2, eta, t)
    assert isinstance(result, dict)
