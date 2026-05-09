"""Tests for spackf.schabenberger_autocorrelation_function."""
import numpy as np
import pytest
from moirais.fn.spackf import schabenberger_autocorrelation_function


def test_spackf_basic():
    """Test basic functionality."""
    x = np.arange(10, dtype=float)
    y = x * 2 + 1
    result = schabenberger_autocorrelation_function(x, y)
    assert abs(result['statistic'] - 1.0) < 0.01


def test_spackf_edge():
    """Test edge cases."""
    result = schabenberger_autocorrelation_function(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
    assert result['n'] == 2
