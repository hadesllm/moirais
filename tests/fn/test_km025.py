"""Tests for km025.kamath_ch2_rts_loss."""
import numpy as np
import pytest
from morie.fn.km025 import kamath_ch2_rts_loss


def test_km025_basic():
    """Test basic functionality."""
    xhat = np.random.default_rng(42).normal(0, 1, 100)
    d = 5
    result = kamath_ch2_rts_loss(xhat, d)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km025_edge():
    """Test edge cases."""
    xhat = np.random.default_rng(42).normal(0, 1, 100)
    d = 5
    result = kamath_ch2_rts_loss(xhat, d)
    assert isinstance(result, dict)
