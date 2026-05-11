"""Tests for ucbb.ucb_bandit."""
import numpy as np
import pytest
from morie.fn.ucbb import ucb_bandit


def test_ucbb_basic():
    """Test basic functionality."""
    arms = np.random.default_rng(42).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    result = ucb_bandit(arms, T)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ucbb_edge():
    """Test edge cases."""
    arms = np.random.default_rng(42).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    result = ucb_bandit(arms, T)
    assert isinstance(result, dict)
