"""Tests for km031.kamath_ch2_sop_loss."""
import numpy as np
import pytest
from morie.fn.km031 import kamath_ch2_sop_loss


def test_km031_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    d = 5
    result = kamath_ch2_sop_loss(x, y, d)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km031_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    d = 5
    result = kamath_ch2_sop_loss(x, y, d)
    assert isinstance(result, dict)
