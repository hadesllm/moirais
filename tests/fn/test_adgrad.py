"""Tests for adgrad.adagrad."""
import numpy as np
import pytest
from moirais.fn.adgrad import adagrad


def test_adgrad_basic():
    """Test basic functionality."""
    g = np.random.default_rng(43).normal(0, 1, 100)
    lr = np.random.default_rng(42).normal(0, 1, 100)
    eps = np.random.default_rng(42).normal(0, 1, 100)
    result = adagrad(g, lr, eps)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_adgrad_edge():
    """Test edge cases."""
    g = np.random.default_rng(43).normal(0, 1, 100)
    lr = np.random.default_rng(42).normal(0, 1, 100)
    eps = np.random.default_rng(42).normal(0, 1, 100)
    result = adagrad(g, lr, eps)
    assert isinstance(result, dict)
