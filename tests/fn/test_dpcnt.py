"""Tests for dpcnt.dp_count."""
import numpy as np
import pytest
from moirais.fn.dpcnt import dp_count


def test_dpcnt_basic():
    """Test basic functionality."""
    D = np.random.default_rng(42).normal(0, 1, 100)
    epsilon = 1e-6
    result = dp_count(D, epsilon)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_dpcnt_edge():
    """Test edge cases."""
    D = np.random.default_rng(42).normal(0, 1, 100)
    epsilon = 1e-6
    result = dp_count(D, epsilon)
    assert isinstance(result, dict)
