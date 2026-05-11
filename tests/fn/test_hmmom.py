"""Tests for hmmom.geron_momentum."""
import numpy as np
import pytest
from morie.fn.hmmom import geron_momentum


def test_hmmom_basic():
    """Test basic functionality."""
    grads = np.random.default_rng(42).normal(0, 1, 100)
    v = np.random.default_rng(44).normal(0, 1, 100)
    beta = 0.8
    eta = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_momentum(grads, v, beta, eta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmmom_edge():
    """Test edge cases."""
    grads = np.random.default_rng(42).normal(0, 1, 100)
    v = np.random.default_rng(44).normal(0, 1, 100)
    beta = 0.8
    eta = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_momentum(grads, v, beta, eta)
    assert isinstance(result, dict)
