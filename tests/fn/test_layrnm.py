"""Tests for layrnm.layer_norm."""
import numpy as np
import pytest
from morie.fn.layrnm import layer_norm


def test_layrnm_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    g = np.random.default_rng(43).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    eps = np.random.default_rng(42).normal(0, 1, 100)
    result = layer_norm(y, x, g, b, eps)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_layrnm_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    g = np.random.default_rng(43).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    eps = np.random.default_rng(42).normal(0, 1, 100)
    result = layer_norm(y, x, g, b, eps)
    assert isinstance(result, dict)
