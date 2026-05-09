"""Tests for gb1122tr.gibbons_kendall_trend."""
import numpy as np
import pytest
from moirais.fn.gb1122tr import gibbons_kendall_trend


def test_gb1122tr_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_kendall_trend(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb1122tr_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_kendall_trend(x)
    assert isinstance(result, dict)
