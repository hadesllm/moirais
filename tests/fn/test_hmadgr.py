"""Tests for hmadgr.geron_adagrad."""
import numpy as np
import pytest
from morie.fn.hmadgr import geron_adagrad


def test_hmadgr_basic():
    """Test basic functionality."""
    grads = np.random.default_rng(42).normal(0, 1, 100)
    s = 90
    eta = np.random.default_rng(42).normal(0, 1, 100)
    eps = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_adagrad(grads, s, eta, eps)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmadgr_edge():
    """Test edge cases."""
    grads = np.random.default_rng(42).normal(0, 1, 100)
    s = 90
    eta = np.random.default_rng(42).normal(0, 1, 100)
    eps = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_adagrad(grads, s, eta, eps)
    assert isinstance(result, dict)
