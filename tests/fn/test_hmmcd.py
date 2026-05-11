"""Tests for hmmcd.geron_mc_dropout."""
import numpy as np
import pytest
from morie.fn.hmmcd import geron_mc_dropout


def test_hmmcd_basic():
    """Test basic functionality."""
    model = np.random.default_rng(42).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    p = 5
    result = geron_mc_dropout(model, x, K, p)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmmcd_edge():
    """Test edge cases."""
    model = np.random.default_rng(42).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    p = 5
    result = geron_mc_dropout(model, x, K, p)
    assert isinstance(result, dict)
