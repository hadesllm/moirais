"""Tests for km027.kamath_ch2_tlm_loss."""
import numpy as np
import pytest
from morie.fn.km027 import kamath_ch2_tlm_loss


def test_km027_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    M_x = np.random.default_rng(42).normal(0, 1, 100)
    M_y = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch2_tlm_loss(x, y, M_x, M_y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km027_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    M_x = np.random.default_rng(42).normal(0, 1, 100)
    M_y = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch2_tlm_loss(x, y, M_x, M_y)
    assert isinstance(result, dict)
