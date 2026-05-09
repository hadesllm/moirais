"""Tests for km039.kamath_ch2_moe_output."""
import numpy as np
import pytest
from moirais.fn.km039 import kamath_ch2_moe_output


def test_km039_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    G = np.eye(10)
    E_i = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch2_moe_output(x, G, E_i)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km039_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    G = np.eye(10)
    E_i = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch2_moe_output(x, G, E_i)
    assert isinstance(result, dict)
