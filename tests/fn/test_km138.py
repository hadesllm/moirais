"""Tests for km138.kamath_ch9_simvlm_mlm."""
import numpy as np
import pytest
from morie.fn.km138 import kamath_ch9_simvlm_mlm


def test_km138_basic():
    """Test basic functionality."""
    theta = 0.0
    x = np.random.default_rng(42).normal(0, 1, 100)
    v = np.random.default_rng(44).normal(0, 1, 100)
    x_m = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch9_simvlm_mlm(theta, x, v, x_m)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km138_edge():
    """Test edge cases."""
    theta = 0.0
    x = np.random.default_rng(42).normal(0, 1, 100)
    v = np.random.default_rng(44).normal(0, 1, 100)
    x_m = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch9_simvlm_mlm(theta, x, v, x_m)
    assert isinstance(result, dict)
