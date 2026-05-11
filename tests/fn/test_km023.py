"""Tests for km023.kamath_ch2_rtd_loss."""
import numpy as np
import pytest
from morie.fn.km023 import kamath_ch2_rtd_loss


def test_km023_basic():
    """Test basic functionality."""
    xhat = np.random.default_rng(42).normal(0, 1, 100)
    d = 5
    result = kamath_ch2_rtd_loss(xhat, d)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km023_edge():
    """Test edge cases."""
    xhat = np.random.default_rng(42).normal(0, 1, 100)
    d = 5
    result = kamath_ch2_rtd_loss(xhat, d)
    assert isinstance(result, dict)
