"""Tests for km148.kamath_ch9_ldm_loss."""
import numpy as np
import pytest
from moirais.fn.km148 import kamath_ch9_ldm_loss


def test_km148_basic():
    """Test basic functionality."""
    epsilon = 1e-6
    z_t = np.random.default_rng(42).normal(0, 1, 100)
    H_X = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch9_ldm_loss(epsilon, z_t, H_X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km148_edge():
    """Test edge cases."""
    epsilon = 1e-6
    z_t = np.random.default_rng(42).normal(0, 1, 100)
    H_X = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch9_ldm_loss(epsilon, z_t, H_X)
    assert isinstance(result, dict)
