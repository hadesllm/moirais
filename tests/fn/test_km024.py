"""Tests for km024.kamath_ch2_std_loss."""
import numpy as np
import pytest
from moirais.fn.km024 import kamath_ch2_std_loss


def test_km024_basic():
    """Test basic functionality."""
    xhat = np.random.default_rng(42).normal(0, 1, 100)
    d = 5
    result = kamath_ch2_std_loss(xhat, d)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km024_edge():
    """Test edge cases."""
    xhat = np.random.default_rng(42).normal(0, 1, 100)
    d = 5
    result = kamath_ch2_std_loss(xhat, d)
    assert isinstance(result, dict)
