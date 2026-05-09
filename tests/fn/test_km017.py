"""Tests for km017.kamath_ch2_ffn_relu."""
import numpy as np
import pytest
from moirais.fn.km017 import kamath_ch2_ffn_relu


def test_km017_basic():
    """Test basic functionality."""
    z = np.random.default_rng(44).normal(0, 1, 100)
    W_1 = np.random.default_rng(42).normal(0, 1, 100)
    W_2 = np.random.default_rng(42).normal(0, 1, 100)
    b_1 = np.random.default_rng(42).normal(0, 1, 100)
    b_2 = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch2_ffn_relu(z, W_1, W_2, b_1, b_2)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km017_edge():
    """Test edge cases."""
    z = np.random.default_rng(44).normal(0, 1, 100)
    W_1 = np.random.default_rng(42).normal(0, 1, 100)
    W_2 = np.random.default_rng(42).normal(0, 1, 100)
    b_1 = np.random.default_rng(42).normal(0, 1, 100)
    b_2 = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch2_ffn_relu(z, W_1, W_2, b_1, b_2)
    assert isinstance(result, dict)
