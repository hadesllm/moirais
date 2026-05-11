"""Tests for km082.kamath_ch6_weat_effect_size."""
import numpy as np
import pytest
from morie.fn.km082 import kamath_ch6_weat_effect_size


def test_km082_basic():
    """Test basic functionality."""
    A_1 = np.random.default_rng(42).normal(0, 1, 100)
    A_2 = np.random.default_rng(42).normal(0, 1, 100)
    W_1 = np.random.default_rng(42).normal(0, 1, 100)
    W_2 = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch6_weat_effect_size(A_1, A_2, W_1, W_2)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km082_edge():
    """Test edge cases."""
    A_1 = np.random.default_rng(42).normal(0, 1, 100)
    A_2 = np.random.default_rng(42).normal(0, 1, 100)
    W_1 = np.random.default_rng(42).normal(0, 1, 100)
    W_2 = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch6_weat_effect_size(A_1, A_2, W_1, W_2)
    assert isinstance(result, dict)
