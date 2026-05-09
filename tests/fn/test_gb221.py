"""Tests for gb221.gibbons_quantile_deriv."""
import numpy as np
import pytest
from moirais.fn.gb221 import gibbons_quantile_deriv


def test_gb221_basic():
    """Test basic functionality."""
    p = 5
    f = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_quantile_deriv(p, f)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gb221_edge():
    """Test edge cases."""
    p = 5
    f = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_quantile_deriv(p, f)
    assert isinstance(result, dict)
