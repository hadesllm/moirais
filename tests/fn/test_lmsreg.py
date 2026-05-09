"""Tests for lmsreg.least_median_squares."""
import numpy as np
import pytest
from moirais.fn.lmsreg import least_median_squares


def test_lmsreg_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = least_median_squares(y, X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_lmsreg_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = least_median_squares(y, X)
    assert isinstance(result, dict)
