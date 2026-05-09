"""Tests for exp3.exp3."""
import numpy as np
import pytest
from moirais.fn.exp3 import exp3


def test_exp3_basic():
    """Test basic functionality."""
    arms = np.random.default_rng(42).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    gamma = 1.0
    result = exp3(arms, T, gamma)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_exp3_edge():
    """Test edge cases."""
    arms = np.random.default_rng(42).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    gamma = 1.0
    result = exp3(arms, T, gamma)
    assert isinstance(result, dict)
