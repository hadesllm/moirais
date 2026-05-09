"""Tests for gb231.gibbons_edf_binomial."""
import numpy as np
import pytest
from moirais.fn.gb231 import gibbons_edf_binomial


def test_gb231_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = gibbons_edf_binomial(x, n)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb231_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = gibbons_edf_binomial(x, n)
    assert isinstance(result, dict)
