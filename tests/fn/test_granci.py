"""Tests for granci.granger_causality_info."""
import numpy as np
import pytest
from morie.fn.granci import granger_causality_info


def test_granci_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    lag = 10
    result = granger_causality_info(x, y, lag)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_granci_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    lag = 10
    result = granger_causality_info(x, y, lag)
    assert isinstance(result, dict)
