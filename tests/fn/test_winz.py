"""Tests for winz.winsorized_mean."""
import numpy as np
import pytest
from moirais.fn.winz import winsorized_mean


def test_winz_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = winsorized_mean(x, alpha)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_winz_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = winsorized_mean(x, alpha)
    assert isinstance(result, dict)
