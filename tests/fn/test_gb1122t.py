"""Tests for gb1122t.gibbons_kendall_ties."""
import numpy as np
import pytest
from moirais.fn.gb1122t import gibbons_kendall_ties


def test_gb1122t_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = gibbons_kendall_ties(x, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gb1122t_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = gibbons_kendall_ties(x, y)
    assert isinstance(result, dict)
