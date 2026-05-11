"""Tests for km096.kamath_ch6_gender_projection_reg."""
import numpy as np
import pytest
from morie.fn.km096 import kamath_ch6_gender_projection_reg


def test_km096_basic():
    """Test basic functionality."""
    W_stereo = np.random.default_rng(42).normal(0, 1, 100)
    g = np.random.default_rng(43).normal(0, 1, 100)
    result = kamath_ch6_gender_projection_reg(W_stereo, g)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km096_edge():
    """Test edge cases."""
    W_stereo = np.random.default_rng(42).normal(0, 1, 100)
    g = np.random.default_rng(43).normal(0, 1, 100)
    result = kamath_ch6_gender_projection_reg(W_stereo, g)
    assert isinstance(result, dict)
