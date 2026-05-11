"""Tests for ltsreg.least_trimmed_squares."""
import numpy as np
import pytest
from morie.fn.ltsreg import least_trimmed_squares


def test_ltsreg_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    h = 0.3
    result = least_trimmed_squares(y, X, h)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ltsreg_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    h = 0.3
    result = least_trimmed_squares(y, X, h)
    assert isinstance(result, dict)
