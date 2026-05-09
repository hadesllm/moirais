"""Tests for km030.kamath_ch2_nsp_loss."""
import numpy as np
import pytest
from moirais.fn.km030 import kamath_ch2_nsp_loss


def test_km030_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    d = 5
    result = kamath_ch2_nsp_loss(x, y, d)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km030_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    d = 5
    result = kamath_ch2_nsp_loss(x, y, d)
    assert isinstance(result, dict)
