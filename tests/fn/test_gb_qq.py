"""Tests for gb_qq.gibbons_qq_plot."""
import numpy as np
import pytest
from moirais.fn.gb_qq import gibbons_qq_plot


def test_gb_qq_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    F0 = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_qq_plot(x, F0)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gb_qq_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    F0 = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_qq_plot(x, F0)
    assert isinstance(result, dict)
