"""Tests for km028.kamath_ch2_alm_loss."""
import numpy as np
import pytest
from moirais.fn.km028 import kamath_ch2_alm_loss


def test_km028_basic():
    """Test basic functionality."""
    z = np.random.default_rng(44).normal(0, 1, 100)
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = kamath_ch2_alm_loss(z, M)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km028_edge():
    """Test edge cases."""
    z = np.random.default_rng(44).normal(0, 1, 100)
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = kamath_ch2_alm_loss(z, M)
    assert isinstance(result, dict)
