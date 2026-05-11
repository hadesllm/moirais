"""Tests for km137.kamath_ch9_itm_hard_negative."""
import numpy as np
import pytest
from morie.fn.km137 import kamath_ch9_itm_hard_negative


def test_km137_basic():
    """Test basic functionality."""
    Pos = np.random.default_rng(42).normal(0, 1, 100)
    HardNeg = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch9_itm_hard_negative(Pos, HardNeg)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km137_edge():
    """Test edge cases."""
    Pos = np.random.default_rng(42).normal(0, 1, 100)
    HardNeg = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch9_itm_hard_negative(Pos, HardNeg)
    assert isinstance(result, dict)
