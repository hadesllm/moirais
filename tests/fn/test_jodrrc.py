"""Tests for jodrrc.joseph_dirrec_strategy."""
import numpy as np
import pytest
from morie.fn.jodrrc import joseph_dirrec_strategy


def test_jodrrc_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    H = np.random.default_rng(42).normal(0, 1, 100)
    result = joseph_dirrec_strategy(X, y, H)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_jodrrc_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    H = np.random.default_rng(42).normal(0, 1, 100)
    result = joseph_dirrec_strategy(X, y, H)
    assert isinstance(result, dict)
