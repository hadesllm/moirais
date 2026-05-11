"""Tests for grmcd.geron_mc_dropout."""
import numpy as np
import pytest
from morie.fn.grmcd import geron_mc_dropout


def test_grmcd_basic():
    """Test basic functionality."""
    model = np.random.default_rng(42).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    p = 5
    result = geron_mc_dropout(model, x, K, p)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_grmcd_edge():
    """Test edge cases."""
    model = np.random.default_rng(42).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    p = 5
    result = geron_mc_dropout(model, x, K, p)
    assert isinstance(result, dict)
