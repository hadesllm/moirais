"""Tests for km080.kamath_ch6_weat_function."""
import numpy as np
import pytest
from morie.fn.km080 import kamath_ch6_weat_function


def test_km080_basic():
    """Test basic functionality."""
    A_1 = np.random.default_rng(42).normal(0, 1, 100)
    A_2 = np.random.default_rng(42).normal(0, 1, 100)
    W_1 = np.random.default_rng(42).normal(0, 1, 100)
    W_2 = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch6_weat_function(A_1, A_2, W_1, W_2)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km080_edge():
    """Test edge cases."""
    A_1 = np.random.default_rng(42).normal(0, 1, 100)
    A_2 = np.random.default_rng(42).normal(0, 1, 100)
    W_1 = np.random.default_rng(42).normal(0, 1, 100)
    W_2 = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch6_weat_function(A_1, A_2, W_1, W_2)
    assert isinstance(result, dict)
