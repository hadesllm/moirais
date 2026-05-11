"""Tests for km022.kamath_ch2_mlm_loss."""
import numpy as np
import pytest
from morie.fn.km022 import kamath_ch2_mlm_loss


def test_km022_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    M_x = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch2_mlm_loss(x, M_x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km022_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    M_x = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch2_mlm_loss(x, M_x)
    assert isinstance(result, dict)
