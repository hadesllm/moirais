"""Tests for km029.kamath_ch2_sbo_loss."""
import numpy as np
import pytest
from morie.fn.km029 import kamath_ch2_sbo_loss


def test_km029_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    S = np.random.default_rng(42).normal(0, 1, 100)
    p = 5
    result = kamath_ch2_sbo_loss(x, S, p)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km029_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    S = np.random.default_rng(42).normal(0, 1, 100)
    p = 5
    result = kamath_ch2_sbo_loss(x, S, p)
    assert isinstance(result, dict)
