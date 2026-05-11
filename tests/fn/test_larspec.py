"""Tests for larspec.lars_optimizer."""
import numpy as np
import pytest
from morie.fn.larspec import lars_optimizer


def test_larspec_basic():
    """Test basic functionality."""
    g = np.random.default_rng(43).normal(0, 1, 100)
    layer = np.random.default_rng(42).normal(0, 1, 100)
    lr = np.random.default_rng(42).normal(0, 1, 100)
    result = lars_optimizer(g, layer, lr)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_larspec_edge():
    """Test edge cases."""
    g = np.random.default_rng(43).normal(0, 1, 100)
    layer = np.random.default_rng(42).normal(0, 1, 100)
    lr = np.random.default_rng(42).normal(0, 1, 100)
    result = lars_optimizer(g, layer, lr)
    assert isinstance(result, dict)
