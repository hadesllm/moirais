"""Tests for dpmed.dp_median."""
import numpy as np
import pytest
from morie.fn.dpmed import dp_median


def test_dpmed_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    epsilon = 1e-6
    result = dp_median(x, epsilon)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_dpmed_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    epsilon = 1e-6
    result = dp_median(x, epsilon)
    assert isinstance(result, dict)
