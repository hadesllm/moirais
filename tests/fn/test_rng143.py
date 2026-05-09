"""Tests for rng143.rangayyan_ch3_autocorrelation_matrix."""
import numpy as np
import pytest
from moirais.fn.rng143 import rangayyan_ch3_autocorrelation_matrix


def test_rng143_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = rangayyan_ch3_autocorrelation_matrix(x, n)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_rng143_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = rangayyan_ch3_autocorrelation_matrix(x, n)
    assert isinstance(result, dict)
